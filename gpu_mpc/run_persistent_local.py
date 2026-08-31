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

IMPORTANT:
This D1 prototype still materializes one sequential key file containing
all chunks.  It is therefore NOT the final large-N bounded-key pipeline.
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
        help="large-disk parent directory for temporary sequential keys",
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

    # D1 deliberately still accumulates all key chunks on disk.
    # Protect against accidentally trying a competition-scale N.
    if n_chunks > 4 and not args.allow_many_chunks:
        raise RuntimeError(
            f"D1 would create {n_chunks} sequential key chunks. "
            f"Refusing because bounded-key D2 is not implemented yet. "
            f"Use --allow-many-chunks only for deliberate testing."
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
        print("Persistent arbitrary-N D1")
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
        # Persistent dealer: one process per party, all chunks.
        # ------------------------------------------------------------
        for party, gpu in ((0, args.gpu0), (1, args.gpu1)):
            env = base_env(weights)

            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["DDG_DEALER_CHUNK_ROOT"] = str(work_root)
            env["DDG_DEALER_CHUNKS"] = str(n_chunks)
            env["DDG_KEYBUF_CAP_GB"] = str(args.keybuf_cap_gb)

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

            run_to_log(
                cmd,
                env,
                logs / f"dealer_p{party}.log",
                f"persistent dealer P{party}",
            )

        print()
        print("===== DEALER SUMMARY =====")

        for party in (0, 1):
            print_matching(
                logs / f"dealer_p{party}.log",
                (
                    "persistent dealer:",
                    "dealer chunk",
                    "persistent dealer complete",
                ),
            )

        # ------------------------------------------------------------
        # Persistent evaluator: two long-lived parties.
        # ------------------------------------------------------------
        common = base_env(weights)
        common["DDG_EVAL_CHUNK_ROOT"] = str(work_root)
        common["DDG_EVAL_CHUNKS"] = str(n_chunks)

        p0_env = common.copy()
        p0_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu0)

        p1_env = common.copy()
        p1_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu1)

        p0_cmd = [
            str(binary),
            str(args.bw),
            str(args.scale),
            "1",
            "0",
            key_dir_arg,
            str(chunk0),
            str(B),
            args.ip,
        ]

        p1_cmd = [
            str(binary),
            str(args.bw),
            str(args.scale),
            "1",
            "1",
            key_dir_arg,
            str(chunk0),
            str(B),
            args.ip,
        ]

        p0_log = logs / "eval_p0.log"
        p1_log = logs / "eval_p1.log"

        print()
        print("[driver] start persistent evaluator P0")

        p0_file = p0_log.open("w")

        p0 = subprocess.Popen(
            p0_cmd,
            env=p0_env,
            stdout=p0_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        time.sleep(1.0)

        print("[driver] start persistent evaluator P1")

        with p1_log.open("w") as f:
            p1 = subprocess.run(
                p1_cmd,
                env=p1_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )

        p0_rc = p0.wait()
        p0_file.close()

        print(
            f"[driver] evaluator return codes: "
            f"P0={p0_rc} P1={p1.returncode}"
        )

        if p0_rc != 0 or p1.returncode != 0:
            print("\n===== P0 LOG TAIL =====")
            for line in p0_log.read_text(
                errors="ignore"
            ).splitlines()[-100:]:
                print(line)

            print("\n===== P1 LOG TAIL =====")
            for line in p1_log.read_text(
                errors="ignore"
            ).splitlines()[-100:]:
                print(line)

            raise RuntimeError(
                "persistent evaluator failed"
            )

        print()
        print("===== EVALUATOR SUMMARY =====")

        print_matching(
            p0_log,
            (
                "DDGOrcaEval",
                "persistent evaluator:",
                "evaluator chunk",
                "persistent evaluator complete",
                "AFFINITY_GLOBAL",
            ),
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
