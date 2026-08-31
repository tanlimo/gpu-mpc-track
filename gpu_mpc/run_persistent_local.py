#!/usr/bin/env python3
"""
Persistent arbitrary-N local DeepDTAGen MPC runner.

D1 correctness prototype:
    logical N
      -> fixed internal micro-batch B
      -> ceil(N / B) fixed-shape chunks
      -> zero-pad final remainder to B
      -> one persistent dealer process per party
      -> one persistent evaluator process per party
      -> trim padded outputs back to N

D2 bounded-key mode:
    key material is exchanged through one reusable key slot per party.
    Key working storage is therefore O(fixed internal B), not O(logical N).

Current remaining scalability limitation:
    fixed-B input chunk directories are still pre-materialized before
    execution.  This is acceptable for the current validation prototype,
    but is not yet fully streamed input I/O.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from run_chunked_local import inspect_source, copy_range


AFF_GLOBAL_RE = re.compile(
    r"^AFFINITY_GLOBAL\[(\d+)\]=([-+0-9.eE]+)$",
    re.MULTILINE,
)


def make_fixed_chunk(
    src: Path,
    dst: Path,
    offset_samples: int,
    real_batch: int,
    fixed_batch: int,
    layout: dict[str, int],
) -> None:
    """
    Copy real_batch samples and zero-pad every tensor file to fixed_batch.

    Padding represents public dummy zero samples.  Outputs belonging to
    padded samples are discarded by this driver.
    """
    if not (0 < real_batch <= fixed_batch):
        raise ValueError(
            f"invalid real_batch={real_batch}, fixed_batch={fixed_batch}"
        )

    dst.mkdir(parents=True, exist_ok=False)

    for name, stride in layout.items():
        out = dst / name

        copy_range(
            src / name,
            out,
            offset_samples * stride,
            real_batch * stride,
        )

        expected_bytes = fixed_batch * stride

        # Extending a regular file with truncate fills the new range with zeros.
        with out.open("r+b") as f:
            f.truncate(expected_bytes)

        actual = out.stat().st_size
        if actual != expected_bytes:
            raise RuntimeError(
                f"{out}: expected {expected_bytes} bytes, got {actual}"
            )


def base_env(weights: Path) -> dict[str, str]:
    env = os.environ.copy()

    env["DDG_WEIGHTS_BIN"] = str(weights)

    for name in (
        "DDG_SLACK_TRUNC",
        "DDG_LOCAL_TRUNC",
        "DDG_INFERENCE_ITERS",
        "DDG_DEALER_CHUNK_ROOT",
        "DDG_DEALER_CHUNKS",
        "DDG_EVAL_CHUNK_ROOT",
        "DDG_EVAL_CHUNKS",
        "DDG_KEYBUF_CAP_GB",
        "DDG_DEALER_EXTERNAL_KEY_IO",
        "DDG_EVAL_EXTERNAL_KEY_IO",
        "DDG_EVAL_KEY_CHUNK_BYTES",
        "DDG_KEY_SLOT_ROOT",
    ):
        env.pop(name, None)

    return env


def run_to_log(
    cmd: list[str],
    env: dict[str, str],
    log: Path,
    label: str,
) -> None:
    print(f"[driver] start {label}", flush=True)

    with log.open("w") as f:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        print(f"\n===== {label} LOG TAIL =====")
        lines = log.read_text(errors="ignore").splitlines()
        for line in lines[-80:]:
            print(line)

        raise RuntimeError(
            f"{label} failed with return code {proc.returncode}"
        )

    print(f"[driver] finish {label}", flush=True)


def wait_for_marker(
    marker: Path,
    proc: subprocess.Popen,
    log: Path,
    label: str,
    timeout_s: float = 1800.0,
) -> None:
    """Wait for a producer marker while detecting early process failure."""
    deadline = time.monotonic() + timeout_s

    while not marker.exists():
        rc = proc.poll()

        if rc is not None:
            lines = log.read_text(errors="ignore").splitlines()

            print(f"\\n===== {label} LOG TAIL =====")
            for line in lines[-80:]:
                print(line)

            raise RuntimeError(
                f"{label} exited before publishing "
                f"{marker.name}; rc={rc}"
            )

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timeout waiting for {marker}"
            )

        time.sleep(0.05)


def stop_process(proc: subprocess.Popen | None) -> None:
    """Best-effort cleanup for a failed local pipeline run."""
    if proc is None or proc.poll() is not None:
        return

    proc.terminate()

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def print_matching(log: Path, needles: tuple[str, ...]) -> None:
    for line in log.read_text(errors="ignore").splitlines():
        if any(x in line for x in needles):
            print(line)


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "source_dir",
        type=Path,
        help="sample-major contiguous MPC input directory",
    )
    ap.add_argument(
        "weights_bin",
        type=Path,
    )
    ap.add_argument(
        "key_parent",
        type=Path,
        help="parent directory for bounded temporary key slots",
    )

    ap.add_argument(
        "--num-samples",
        "-n",
        type=int,
        required=True,
    )
    ap.add_argument(
        "--micro-batch",
        "-b",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--bw",
        type=int,
        default=64,
    )
    ap.add_argument(
        "--scale",
        type=int,
        default=12,
    )

    ap.add_argument("--gpu0", default="0")
    ap.add_argument("--gpu1", default="1")
    ap.add_argument("--ip", default="127.0.0.1")

    ap.add_argument(
        "--keybuf-cap-gb",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--allow-many-chunks",
        action="store_true",
        help=(
            "allow >4 chunks even though D1 still accumulates "
            "all sequential keys on disk"
        ),
    )
    ap.add_argument("--keep-keys", action="store_true")
    ap.add_argument("--keep-work", action="store_true")

    args = ap.parse_args()

    src = args.source_dir.resolve()
    weights = args.weights_bin.resolve()
    key_parent = args.key_parent.resolve()

    if args.num_samples <= 0:
        ap.error("--num-samples must be > 0")

    if not (1 <= args.micro_batch <= 128):
        ap.error("--micro-batch must be in [1,128]")

    if args.bw not in (32, 64):
        ap.error("--bw must be 32 or 64")

    if not weights.is_file():
        raise FileNotFoundError(weights)

    available, layout = inspect_source(src, args.bw)

    if args.num_samples > available:
        raise RuntimeError(
            f"requested N={args.num_samples}, "
            f"but source contains only {available}"
        )

    B = args.micro_batch
    N = args.num_samples
    n_chunks = (N + B - 1) // B
    padded_n = n_chunks * B

    # D2 bounds key storage, but this validation runner still
    # pre-materializes every fixed-B input chunk directory.
    if n_chunks > 4 and not args.allow_many_chunks:
        raise RuntimeError(
            f"current D2 runner would pre-materialize {n_chunks} "
            f"input chunk directories. Use --allow-many-chunks only "
            f"for deliberate larger validation runs."
        )

    repo = Path(__file__).resolve().parent.parent
    binary = repo / "gpu_mpc" / "deepdtagen_inference"

    if not binary.is_file():
        raise FileNotFoundError(
            f"missing binary: {binary}; build BW={args.bw} first"
        )

    key_parent.mkdir(parents=True, exist_ok=True)

    work_root = Path(
        tempfile.mkdtemp(
            prefix="deepdtagen_persistent_chunks_",
            dir="/tmp",
        )
    )

    key_run = Path(
        tempfile.mkdtemp(
            prefix="deepdtagen_persistent_keys_",
            dir=key_parent,
        )
    )

    logs = work_root / "logs"
    logs.mkdir()

    success = False

    try:
        print("========================================")
        print("Persistent arbitrary-N D2 bounded-key")
        print("========================================")
        print(f"source           = {src}")
        print(f"available N      = {available}")
        print(f"requested N      = {N}")
        print(f"internal B       = {B}")
        print(f"chunks           = {n_chunks}")
        print(f"padded N         = {padded_n}")
        print(f"padding samples  = {padded_n - N}")
        print(f"BW / SCALE       = {args.bw} / {args.scale}")
        print(f"work root        = {work_root}")
        print(f"key run          = {key_run}")

        # ------------------------------------------------------------
        # Prepare fixed-B chunk directories.
        # ------------------------------------------------------------
        offset = 0

        for chunk in range(n_chunks):
            real_batch = min(B, N - offset)

            chunk_dir = work_root / f"chunk_{chunk:05d}"

            make_fixed_chunk(
                src=src,
                dst=chunk_dir,
                offset_samples=offset,
                real_batch=real_batch,
                fixed_batch=B,
                layout=layout,
            )

            print(
                f"[driver] chunk={chunk} "
                f"real={real_batch} padded={B - real_batch} "
                f"global={offset}..{offset + real_batch - 1}"
            )

            offset += real_batch

        chunk0 = work_root / "chunk_00000"

        key_dir_arg = str(key_run) + "/"

        # ------------------------------------------------------------
        # D2 bounded single-slot key pipeline.
        #
        # Dealer P0/P1 are persistent producers.
        # Evaluator P0/P1 are persistent consumers.
        #
        # Each party owns exactly one reusable large key slot:
        #
        #   Dealer -> partyX.key -> Evaluator -> ACK -> reuse
        #
        # No O(number_of_chunks) sequential key file is created.
        # ------------------------------------------------------------
        slot_root = key_run

        key_stub = work_root / "key_stub"
        key_stub.mkdir()

        key_dir_arg = str(key_stub) + "/"

        dealer_procs: dict[int, subprocess.Popen] = {}
        dealer_files = {}

        eval_procs: dict[int, subprocess.Popen] = {}
        eval_files = {}

        try:
            # --------------------------------------------------------
            # Start BOTH persistent Dealers.
            #
            # They generate chunk 0, publish the slot, then block
            # waiting for the corresponding evaluator ACK.
            # --------------------------------------------------------
            for party, gpu in (
                (0, args.gpu0),
                (1, args.gpu1),
            ):
                env = base_env(weights)

                env["CUDA_VISIBLE_DEVICES"] = str(gpu)

                env["DDG_DEALER_CHUNK_ROOT"] = str(
                    work_root
                )
                env["DDG_DEALER_CHUNKS"] = str(
                    n_chunks
                )
                env["DDG_KEYBUF_CAP_GB"] = str(
                    args.keybuf_cap_gb
                )

                env["DDG_DEALER_EXTERNAL_KEY_IO"] = "1"
                env["DDG_KEY_SLOT_ROOT"] = str(
                    slot_root
                )

                cmd = [
                    str(binary),
                    str(args.bw),
                    str(args.scale),
                    "0",
                    str(party),
                    key_dir_arg,
                    str(chunk0),
                    str(B),
                ]

                log = logs / f"dealer_p{party}.log"

                print(
                    f"[driver] start bounded dealer P{party}",
                    flush=True,
                )

                fh = log.open("w")

                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                dealer_files[party] = fh
                dealer_procs[party] = proc

            # --------------------------------------------------------
            # Wait until BOTH first slots are completely written.
            #
            # This lets us detect key_chunk_bytes rather than
            # hard-coding the B=8 value 2810019840.
            # --------------------------------------------------------
            for party in (0, 1):
                ready = (
                    slot_root /
                    f"party{party}.ready.00000"
                )

                wait_for_marker(
                    marker=ready,
                    proc=dealer_procs[party],
                    log=logs / f"dealer_p{party}.log",
                    label=f"dealer P{party}",
                )

            slot0 = slot_root / "party0.key"
            slot1 = slot_root / "party1.key"

            if not slot0.is_file():
                raise RuntimeError(
                    f"missing first P0 key slot: {slot0}"
                )

            if not slot1.is_file():
                raise RuntimeError(
                    f"missing first P1 key slot: {slot1}"
                )

            key_bytes0 = slot0.stat().st_size
            key_bytes1 = slot1.stat().st_size

            if key_bytes0 != key_bytes1:
                raise RuntimeError(
                    "party key chunk sizes differ: "
                    f"P0={key_bytes0} "
                    f"P1={key_bytes1}"
                )

            if (
                key_bytes0 <= 0 or
                key_bytes0 % 4096 != 0
            ):
                raise RuntimeError(
                    f"invalid key chunk bytes="
                    f"{key_bytes0}"
                )

            key_chunk_bytes = key_bytes0

            print(
                "[driver] detected key chunk bytes = "
                f"{key_chunk_bytes}",
                flush=True,
            )

            # --------------------------------------------------------
            # Start BOTH persistent Evaluators.
            #
            # keySize comes from the detected fixed-B slot size.
            # DDG_EVAL_EXTERNAL_KEY_IO means the evaluator constructor
            # does NOT require a pre-existing sequential key file.
            # --------------------------------------------------------
            for party, gpu in (
                (0, args.gpu0),
                (1, args.gpu1),
            ):
                env = base_env(weights)

                env["CUDA_VISIBLE_DEVICES"] = str(gpu)

                env["DDG_EVAL_CHUNK_ROOT"] = str(
                    work_root
                )
                env["DDG_EVAL_CHUNKS"] = str(
                    n_chunks
                )

                env["DDG_EVAL_KEY_CHUNK_BYTES"] = str(
                    key_chunk_bytes
                )
                env["DDG_EVAL_EXTERNAL_KEY_IO"] = "1"
                env["DDG_KEY_SLOT_ROOT"] = str(
                    slot_root
                )

                cmd = [
                    str(binary),
                    str(args.bw),
                    str(args.scale),
                    "1",
                    str(party),
                    key_dir_arg,
                    str(chunk0),
                    str(B),
                    args.ip,
                ]

                log = logs / f"eval_p{party}.log"

                # P0 normally listens; start it first.
                if party == 1:
                    time.sleep(1.0)

                print(
                    f"[driver] start bounded evaluator P{party}",
                    flush=True,
                )

                fh = log.open("w")

                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                eval_files[party] = fh
                eval_procs[party] = proc

            # --------------------------------------------------------
            # Evaluators consume every slot and ACK every chunk.
            # Dealers therefore advance and reuse the same large files.
            # --------------------------------------------------------
            e0_rc = eval_procs[0].wait()
            e1_rc = eval_procs[1].wait()

            eval_files[0].close()
            eval_files[1].close()

            d0_rc = dealer_procs[0].wait()
            d1_rc = dealer_procs[1].wait()

            dealer_files[0].close()
            dealer_files[1].close()

            print(
                "[driver] return codes: "
                f"D0={d0_rc} D1={d1_rc} "
                f"E0={e0_rc} E1={e1_rc}"
            )

            if any(
                rc != 0
                for rc in (
                    d0_rc,
                    d1_rc,
                    e0_rc,
                    e1_rc,
                )
            ):
                for name in (
                    "dealer_p0.log",
                    "dealer_p1.log",
                    "eval_p0.log",
                    "eval_p1.log",
                ):
                    log = logs / name

                    print(
                        f"\n===== {name} TAIL ====="
                    )

                    for line in log.read_text(
                        errors="ignore"
                    ).splitlines()[-100:]:
                        print(line)

                raise RuntimeError(
                    "bounded persistent pipeline failed"
                )

        except Exception:
            # Avoid leaving a Dealer blocked for 30 minutes waiting
            # for an ACK if a local validation run fails.
            for proc in eval_procs.values():
                stop_process(proc)

            for proc in dealer_procs.values():
                stop_process(proc)

            for fh in eval_files.values():
                if not fh.closed:
                    fh.close()

            for fh in dealer_files.values():
                if not fh.closed:
                    fh.close()

            raise

        print()
        print("===== DEALER SUMMARY =====")

        for party in (0, 1):
            print_matching(
                logs / f"dealer_p{party}.log",
                (
                    "bounded-key mode",
                    "dealer chunk",
                    "slot ready",
                    "slot acked",
                    "persistent dealer complete",
                ),
            )

        print()
        print("===== EVALUATOR SUMMARY =====")

        p0_log = logs / "eval_p0.log"

        print_matching(
            p0_log,
            (
                "explicit key chunk",
                "bounded-key mode",
                "evaluator chunk",
                "slot loaded",
                "slot ack",
                "persistent evaluator complete",
                "AFFINITY_GLOBAL",
            ),
        )

        # After the last evaluator ACK, the Dealer removes the
        # reusable large slot and its ready/ack markers.
        leftovers = [
            x
            for x in slot_root.iterdir()
            if x.is_file()
        ]

        if leftovers:
            raise RuntimeError(
                "key slot root is not empty after success: "
                + ", ".join(
                    f"{x.name}="
                    f"{x.stat().st_size}"
                    for x in leftovers
                )
            )

        print(
            "[driver] bounded key slot cleanup: PASS"
        )

        # ------------------------------------------------------------
        # Parse padded outputs, then trim to logical N.
        # ------------------------------------------------------------
        text = p0_log.read_text(errors="ignore")

        pairs = AFF_GLOBAL_RE.findall(text)

        out_map = {
            int(i): float(v)
            for i, v in pairs
        }

        expected_padded = set(range(padded_n))
        actual = set(out_map)

        if actual != expected_padded:
            raise RuntimeError(
                f"expected padded global indices "
                f"0..{padded_n - 1}, got {sorted(actual)}"
            )

        padded_outputs = [
            out_map[i]
            for i in range(padded_n)
        ]

        outputs = padded_outputs[:N]

        print()
        print("========================================")
        print("LOGICAL RESULTS")
        print("========================================")

        for i, value in enumerate(outputs):
            print(
                f"AFFINITY_GLOBAL[{i}]={value:.6f}"
            )

        if padded_n != N:
            print()
            print(
                f"[driver] trimmed {padded_n - N} "
                f"padding output(s)"
            )

        print()
        print(
            f"PASS: N={N}, fixed B={B}, "
            f"chunks={n_chunks}, padded={padded_n}, "
            f"returned={len(outputs)}"
        )

        success = True
        return 0

    finally:
        if success:
            if not args.keep_keys:
                shutil.rmtree(
                    key_run,
                    ignore_errors=True,
                )

            if not args.keep_work:
                shutil.rmtree(
                    work_root,
                    ignore_errors=True,
                )
        else:
            print()
            print("[driver] FAILURE: retaining artifacts")
            print(f"[driver] work = {work_root}")
            print(f"[driver] keys = {key_run}")


if __name__ == "__main__":
    raise SystemExit(main())
