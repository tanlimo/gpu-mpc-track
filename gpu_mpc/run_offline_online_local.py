#!/usr/bin/env python3
"""
Local two-GPU online test using PRE-GENERATED offline FSS keys.

Lifecycle:

    OFFLINE, already completed:
        one logical Dealer
          -> complete K0
          -> complete K1
          -> Dealer exits

    THIS SCRIPT:
        materialize fixed-B input chunks
        timed public Protein GatedCNN
        start Evaluator P0 on GPU0
        start Evaluator P1 on GPU1
        consume complete sequential K0/K1 files
        run 2PC
        trim padded outputs

This script NEVER starts role=0 / Dealer processes.

Current baseline mode:
    offline full key files + online sequential file reads.

The next stage will preload the complete party key into CPU RAM before
ONLINE timing starts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from run_chunked_local import inspect_source, copy_range

from run_persistent_local import (
    AFF_GLOBAL_RE,
    EVAL_PROFILE_RE,
    base_env,
    make_fixed_chunk,
    parse_profile_totals,
    stop_process,
)


def wait_for_path(
    path: Path,
    timeout: float = 600.0,
):
    deadline = time.monotonic() + timeout

    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timeout waiting for {path}"
            )

        time.sleep(0.01)



def load_offline_metadata(
    key_dir: Path,
) -> dict:
    path = key_dir / "metadata.json"

    if not path.is_file():
        raise FileNotFoundError(
            f"missing offline Dealer metadata: {path}"
        )

    with path.open() as f:
        meta = json.load(f)

    if meta.get("mode") != "offline_full_key_file":
        raise RuntimeError(
            "unsupported offline key mode: "
            f"{meta.get('mode')!r}"
        )

    if meta.get("logical_dealer_count") != 1:
        raise RuntimeError(
            "offline metadata does not describe "
            "one logical Dealer"
        )

    if meta.get("online_evaluator_started") is not False:
        raise RuntimeError(
            "offline metadata unexpectedly says an "
            "online evaluator was started"
        )

    return meta


def validate_offline_keys(
    *,
    key_dir: Path,
    meta: dict,
    samples: int,
    batch: int,
    chunks: int,
    bw: int,
    scale: int,
    secure_adj_norm: bool,
) -> tuple[Path, Path, int]:
    expected = {
        "samples": samples,
        "micro_batch": batch,
        "chunks": chunks,
        "bw": bw,
        "scale": scale,
        "secure_adj_norm": secure_adj_norm,
    }

    for name, value in expected.items():
        actual = meta.get(name)

        if actual != value:
            raise RuntimeError(
                f"offline key metadata mismatch for {name}: "
                f"expected={value!r}, actual={actual!r}"
            )

    files = meta.get("key_files")

    if not isinstance(files, dict):
        raise RuntimeError(
            "offline metadata missing key_files"
        )

    try:
        k0 = key_dir / files["party0"]
        k1 = key_dir / files["party1"]
    except KeyError as e:
        raise RuntimeError(
            f"offline metadata missing {e.args[0]}"
        ) from e

    for party, path in (
        (0, k0),
        (1, k1),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"missing K{party}: {path}"
            )

    bytes0 = k0.stat().st_size
    bytes1 = k1.stat().st_size

    if bytes0 <= 0:
        raise RuntimeError(
            "offline key files are empty"
        )

    if bytes0 != bytes1:
        raise RuntimeError(
            "offline party key sizes differ: "
            f"K0={bytes0}, K1={bytes1}"
        )

    expected_total = meta.get(
        "key_total_bytes_per_party"
    )

    if expected_total != bytes0:
        raise RuntimeError(
            "offline key byte count mismatch: "
            f"metadata={expected_total}, actual={bytes0}"
        )

    if bytes0 % chunks != 0:
        raise RuntimeError(
            "offline key bytes are not divisible by "
            f"chunks={chunks}"
        )

    key_chunk_bytes = bytes0 // chunks

    meta_chunk = meta.get(
        "key_chunk_bytes_per_party"
    )

    if meta_chunk != key_chunk_bytes:
        raise RuntimeError(
            "offline key chunk size mismatch: "
            f"metadata={meta_chunk}, "
            f"actual={key_chunk_bytes}"
        )

    return k0, k1, key_chunk_bytes


def print_log_tail(
    path: Path,
    lines: int = 100,
) -> None:
    if not path.is_file():
        print(f"[missing log] {path}")
        return

    text = path.read_text(
        errors="ignore"
    ).splitlines()

    for line in text[-lines:]:
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run local online DeepDTAGen 2PC using complete "
            "FSS key files generated by run_offline_dealer.py."
        )
    )

    ap.add_argument(
        "source_dir",
        type=Path,
        help="prepared dataset directory",
    )

    ap.add_argument(
        "weights_bin",
        type=Path,
        help="prepared MPC weights.bin",
    )

    ap.add_argument(
        "offline_key_dir",
        type=Path,
        help=(
            "directory produced by "
            "run_offline_dealer.py"
        ),
    )

    ap.add_argument(
        "--protein-checkpoint",
        type=Path,
        required=True,
        help=(
            "released DeepDTAGen .pth containing cnn.*"
        ),
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

    ap.add_argument(
        "--gpu0",
        default="0",
    )

    ap.add_argument(
        "--gpu1",
        default="1",
    )

    ap.add_argument(
        "--timeout",
        type=float,
        default=600.0,
    )

    ap.add_argument(
        "--ip",
        default="127.0.0.1",
        help=(
            "GpuPeer address. Local two-GPU test uses "
            "127.0.0.1; current backend port is 42003."
        ),
    )

    ap.add_argument(
        "--legacy-precomputed-adj",
        action="store_true",
    )

    ap.add_argument(
        "--full-key-ram",
        action="store_true",
        help=(
            "preload the complete party FSS key file directly "
            "into the final evaluator key buffer"
        ),
    )

    ap.add_argument(
        "--keep-work",
        action="store_true",
    )

    args = ap.parse_args()

    src = args.source_dir.resolve()
    weights = args.weights_bin.resolve()
    key_dir = args.offline_key_dir.resolve()
    protein_checkpoint = (
        args.protein_checkpoint.resolve()
    )

    if args.num_samples <= 0:
        ap.error("--num-samples must be > 0")

    if not (1 <= args.micro_batch <= 128):
        ap.error("--micro-batch must be in [1,128]")

    if args.bw not in (32, 64):
        ap.error("--bw must be 32 or 64")

    if not src.is_dir():
        raise FileNotFoundError(src)

    if not weights.is_file():
        raise FileNotFoundError(weights)

    if not key_dir.is_dir():
        raise FileNotFoundError(key_dir)

    if not protein_checkpoint.is_file():
        raise FileNotFoundError(
            protein_checkpoint
        )

    # No protein_emb.dat is required from preprocessing.
    available, layout = inspect_source(
        src,
        args.bw,
        require_protein_emb=False,
    )

    if args.num_samples > available:
        raise RuntimeError(
            f"requested N={args.num_samples}, "
            f"but source contains only {available}"
        )

    target_ids_src = (
        src /
        "target_ids.dat"
    )

    if not target_ids_src.is_file():
        raise FileNotFoundError(
            f"timed Protein requires {target_ids_src}"
        )

    target_stride = 1000 * 8

    if (
        target_ids_src.stat().st_size %
        target_stride != 0
    ):
        raise RuntimeError(
            "invalid target_ids.dat size"
        )

    target_samples = (
        target_ids_src.stat().st_size //
        target_stride
    )

    if target_samples != available:
        raise RuntimeError(
            "target_ids sample count mismatch: "
            f"target={target_samples}, "
            f"MPC={available}"
        )

    N = args.num_samples
    B = args.micro_batch

    chunks = (N + B - 1) // B
    padded_n = chunks * B

    secure_adj_norm = (
        not args.legacy_precomputed_adj
    )

    meta = load_offline_metadata(
        key_dir
    )

    k0, k1, key_chunk_bytes = (
        validate_offline_keys(
            key_dir=key_dir,
            meta=meta,
            samples=N,
            batch=B,
            chunks=chunks,
            bw=args.bw,
            scale=args.scale,
            secure_adj_norm=secure_adj_norm,
        )
    )

    repo = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    binary = (
        repo /
        "gpu_mpc" /
        "deepdtagen_inference"
    )

    protein_worker = (
        repo /
        "reference" /
        "run_protein_chunks.py"
    )

    if not binary.is_file():
        raise FileNotFoundError(
            f"missing binary: {binary}"
        )

    if not protein_worker.is_file():
        raise FileNotFoundError(
            protein_worker
        )

    work_root = Path(
        tempfile.mkdtemp(
            prefix=(
                "deepdtagen_offline_online_chunks_"
            ),
            dir="/tmp",
        )
    )

    logs = (
        work_root /
        "logs"
    )

    logs.mkdir()

    success = False

    eval_procs: dict[
        int,
        subprocess.Popen,
    ] = {}

    eval_files = {}

    try:
        print(
            "========================================"
        )
        print(
            "DeepDTAGen Offline-Key Online Local Test"
        )
        print(
            "========================================"
        )

        print(f"source          = {src}")
        print(f"weights         = {weights}")
        print(f"offline keys    = {key_dir}")
        print(f"K0              = {k0}")
        print(f"K1              = {k1}")
        print(f"logical N       = {N}")
        print(f"micro-batch B   = {B}")
        print(f"chunks          = {chunks}")
        print(f"padded N        = {padded_n}")
        print(
            f"BW / SCALE      = "
            f"{args.bw} / {args.scale}"
        )
        print(
            f"GPU P0 / P1     = "
            f"{args.gpu0} / {args.gpu1}"
        )
        print(f"peer IP         = {args.ip}")
        print(
            "peer port       = 42003 "
            "(current GpuPeer default)"
        )
        print(
            "Dealer          = NOT STARTED"
        )
        print(
            "key mode        = " +
            (
                "offline full file -> direct full RAM buffer"
                if args.full_key_ram
                else "offline full file; sequential online reads"
            )
        )
        print(
            "Protein         = timed FP32 GPU"
        )
        print(
            "A_norm          = " +
            (
                "secure online"
                if secure_adj_norm
                else "legacy precomputed"
            )
        )
        print()

        # --------------------------------------------------------
        # Input format materialization is outside online timing.
        # --------------------------------------------------------
        materialize_start_ns = (
            time.perf_counter_ns()
        )

        offset = 0

        for chunk in range(chunks):
            real_batch = min(
                B,
                N - offset,
            )

            chunk_dir = (
                work_root /
                f"chunk_{chunk:05d}"
            )

            make_fixed_chunk(
                src=src,
                dst=chunk_dir,
                offset_samples=offset,
                real_batch=real_batch,
                fixed_batch=B,
                layout=layout,
            )

            target_out = (
                chunk_dir /
                "target_ids.dat"
            )

            copy_range(
                target_ids_src,
                target_out,
                offset * target_stride,
                real_batch * target_stride,
            )

            # Public padding index = 0.
            with target_out.open("r+b") as f:
                f.truncate(
                    B * target_stride
                )

            if (
                target_out.stat().st_size
                != B * target_stride
            ):
                raise RuntimeError(
                    f"{target_out}: invalid size"
                )

            print(
                f"[driver] chunk={chunk} "
                f"real={real_batch} "
                f"padded={B - real_batch} "
                f"global={offset}.."
                f"{offset + real_batch - 1}"
            )

            offset += real_batch

        materialize_end_ns = (
            time.perf_counter_ns()
        )

        materialize_us = (
            materialize_end_ns -
            materialize_start_ns
        ) // 1000

        chunk0 = (
            work_root /
            "chunk_00000"
        )

        # --------------------------------------------------------
        # Evaluator SETUP.
        #
        # This phase is intentionally outside ONLINE_COMPUTE:
        #   * evaluator process launch
        #   * CUDA/backend initialization
        #   * full FSS key preload
        #   * peer connection
        #   * READY synchronization
        #
        # It is still reported separately and is included in
        # END_TO_END.  PRE_COMPLIANCE remains in effect because
        # official treatment of model-specific FSS preprocessing
        # is not assumed here.
        # --------------------------------------------------------
        key_dir_arg = (
            str(key_dir) +
            "/"
        )

        ready_files = {
            0: work_root / "eval_p0.ready",
            1: work_root / "eval_p1.ready",
        }

        start_file = (
            work_root /
            "evaluators.start"
        )

        for path in (
            ready_files[0],
            ready_files[1],
            start_file,
        ):
            path.unlink(
                missing_ok=True
            )

        setup_start_ns = (
            time.perf_counter_ns()
        )

        end_to_end_start_ns = (
            setup_start_ns
        )

        print(
            "[driver] SETUP: start Evaluators "
            "for full-key preload / peer setup",
            flush=True,
        )

        for party, gpu in (
            (0, args.gpu0),
            (1, args.gpu1),
        ):
            env = base_env(
                weights,
                secure_adj_norm=secure_adj_norm,
            )

            env[
                "CUDA_VISIBLE_DEVICES"
            ] = str(gpu)

            env[
                "DDG_EVAL_CHUNK_ROOT"
            ] = str(work_root)

            env[
                "DDG_EVAL_CHUNKS"
            ] = str(chunks)

            # READY is party-specific; START is shared.
            env[
                "DDG_EVAL_READY_FILE"
            ] = str(
                ready_files[party]
            )

            env[
                "DDG_EVAL_START_FILE"
            ] = str(
                start_file
            )

            if args.full_key_ram:
                env[
                    "DDG_EVAL_FULL_KEY_RAM"
                ] = "1"
            else:
                env.pop(
                    "DDG_EVAL_FULL_KEY_RAM",
                    None,
                )

            # Explicitly guarantee full-file mode rather than
            # the old bounded Dealer/Evaluator streaming path.
            for name in (
                "DDG_EVAL_EXTERNAL_KEY_IO",
                "DDG_KEY_SLOT_ROOT",
                "DDG_EVAL_KEY_CHUNK_BYTES",
            ):
                env.pop(
                    name,
                    None,
                )

            cmd = [
                str(binary),
                str(args.bw),
                str(args.scale),
                "1",              # evaluator only
                str(party),
                key_dir_arg,
                str(chunk0),
                str(B),
                args.ip,
            ]

            log = (
                logs /
                f"eval_p{party}.log"
            )

            # Keep the historical connection-safety delay for now.
            #
            # IMPORTANT:
            # this is SETUP, not ONLINE_COMPUTE.
            #
            # We will remove/reduce it only after confirming the
            # GpuPeer connection implementation safely retries.
            if party == 1:
                time.sleep(1.0)

            print(
                f"[driver] SETUP: start "
                f"Evaluator P{party}",
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
        # Wait until BOTH Evaluators report READY.
        #
        # Unlike the previous experiment, do not blindly wait for
        # marker files.  If either process exits, fail immediately
        # and show its log rather than waiting for a long timeout.
        # --------------------------------------------------------
        ready_deadline = (
            time.monotonic() +
            600.0
        )

        while True:
            ready0 = (
                ready_files[0].exists()
            )

            ready1 = (
                ready_files[1].exists()
            )

            if ready0 and ready1:
                break

            failed_party = None
            failed_rc = None

            for party in (0, 1):
                rc = (
                    eval_procs[party].
                    poll()
                )

                if rc is not None:
                    failed_party = party
                    failed_rc = rc
                    break

            if failed_party is not None:
                # Stop the other evaluator if it is still alive.
                for party in (0, 1):
                    proc = eval_procs[party]

                    if proc.poll() is None:
                        proc.terminate()

                for party in (0, 1):
                    try:
                        eval_procs[party].wait(
                            timeout=5
                        )
                    except subprocess.TimeoutExpired:
                        eval_procs[party].kill()
                        eval_procs[party].wait()

                for party in (0, 1):
                    eval_files[party].close()

                print()
                print(
                    f"[driver] ERROR: Evaluator "
                    f"P{failed_party} exited before READY "
                    f"with rc={failed_rc}"
                )

                for party in (0, 1):
                    print()
                    print(
                        "===== "
                        f"EVALUATOR P{party} LOG TAIL "
                        "====="
                    )

                    print_log_tail(
                        logs /
                        f"eval_p{party}.log"
                    )

                raise RuntimeError(
                    "evaluator exited before READY"
                )

            if (
                time.monotonic() >=
                ready_deadline
            ):
                for party in (0, 1):
                    proc = eval_procs[party]

                    if proc.poll() is None:
                        proc.terminate()

                for party in (0, 1):
                    try:
                        eval_procs[party].wait(
                            timeout=5
                        )
                    except subprocess.TimeoutExpired:
                        eval_procs[party].kill()
                        eval_procs[party].wait()

                for party in (0, 1):
                    eval_files[party].close()

                for party in (0, 1):
                    print()
                    print(
                        "===== "
                        f"EVALUATOR P{party} LOG TAIL "
                        "====="
                    )

                    print_log_tail(
                        logs /
                        f"eval_p{party}.log"
                    )

                raise RuntimeError(
                    "timeout waiting for evaluator READY"
                )

            time.sleep(0.01)

        setup_end_ns = (
            time.perf_counter_ns()
        )

        setup_wall_us = (
            setup_end_ns -
            setup_start_ns
        ) // 1000

        print(
            "[driver] SETUP: both Evaluators READY: "
            f"{setup_wall_us} us",
            flush=True,
        )

        # ========================================================
        # ONLINE COMPUTE START.
        #
        # Everything above is setup.
        #
        # From here onward:
        #   public Protein model
        #   + secure adjacency normalization
        #   + 2PC affinity inference
        # are included.
        # ========================================================
        online_start_ns = (
            time.perf_counter_ns()
        )


        # --------------------------------------------------------
        # Timed public Protein GatedCNN.
        #
        # Persistent worker v1:
        # Python startup, CUDA initialization and checkpoint loading
        # are moved before ONLINE timing.
        # ONLINE only sends command and waits completion.
        # --------------------------------------------------------

        print(
            "[driver] SETUP: starting Protein persistent worker",
            flush=True,
        )

        protein_worker_dir = (
            work_root /
            "protein_worker"
        )

        protein_ready_file = (
            protein_worker_dir /
            "ready"
        )

        protein_done_file = (
            protein_worker_dir /
            "done"
        )

        protein_command_file = (
            protein_worker_dir /
            "command.json"
        )

        protein_worker_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        protein_env = os.environ.copy()

        protein_env[
            "CUDA_VISIBLE_DEVICES"
        ] = str(args.gpu0)


        protein_cmd = [
            sys.executable,
            str(protein_worker),
            "--checkpoint",
            str(protein_checkpoint),
            "--chunk-root",
            str(work_root),
            "--chunks",
            str(chunks),
            "--batch",
            str(B),
            "--scale",
            str(args.scale),
            "--bw",
            str(args.bw),
            "--persistent-worker",
            "--worker-dir",
            str(protein_worker_dir),
        ]


        protein_proc = subprocess.Popen(
            protein_cmd,
            env=protein_env,
            text=True,
        )


        wait_for_path(
            protein_ready_file,
            timeout=args.timeout,
        )


        print(
            "[driver] SETUP: Protein worker READY",
            flush=True,
        )


        print(
            "[driver] ONLINE: start timed public "
            "Protein GatedCNN",
            flush=True,
        )


        protein_start_ns = time.perf_counter_ns()


        protein_done_file.unlink(
            missing_ok=True,
        )

        protein_command_file.unlink(
            missing_ok=True,
        )


        command = {
            "chunk_root": str(work_root),
            "chunks": chunks,
            "batch": B,
        }


        protein_command_file.write_text(
            json.dumps(command)
        )


        wait_for_path(
            protein_done_file,
            timeout=args.timeout,
        )


        protein_end_ns = time.perf_counter_ns()


        protein_runtime_us = (
            protein_end_ns -
            protein_start_ns
        ) // 1000


        print(
            "[driver] ONLINE: Protein complete: "
            f"{protein_runtime_us} us",
            flush=True,
        )

        # --------------------------------------------------------
        # Protein outputs are now present for every chunk.
        # Release both Evaluators simultaneously into MPC.
        # --------------------------------------------------------
        start_file.write_text(
            "1\n"
        )

        print(
            "[driver] ONLINE: release Evaluators "
            "with START marker",
            flush=True,
        )

        e0_rc = eval_procs[0].wait()
        e1_rc = eval_procs[1].wait()

        eval_files[0].close()
        eval_files[1].close()

        online_end_ns = (
            time.perf_counter_ns()
        )

        end_to_end_end_ns = (
            online_end_ns
        )

        online_wall_us = (
            online_end_ns -
            online_start_ns
        ) // 1000

        end_to_end_us = (
            end_to_end_end_ns -
            end_to_end_start_ns
        ) // 1000

        print(
            "[driver] evaluator return codes: "
            f"E0={e0_rc} E1={e1_rc}"
        )

        if e0_rc != 0 or e1_rc != 0:
            for party in (0, 1):
                print()
                print(
                    "===== "
                    f"EVALUATOR P{party} LOG TAIL "
                    "====="
                )

                print_log_tail(
                    logs /
                    f"eval_p{party}.log"
                )

            raise RuntimeError(
                "offline-key online pipeline failed"
            )

        # --------------------------------------------------------
        # Parse direct full-RAM preload metadata.
        #
        # Evaluators are now started during SETUP and reach READY
        # only after backend/peer initialization and, in full-RAM
        # mode, complete direct key preload.
        #
        # Therefore full-RAM preload is outside ONLINE_COMPUTE,
        # but remains visible in SETUP_TOTAL and END_TO_END.
        # --------------------------------------------------------
        preload_info = {}

        preload_re = re.compile(
            r"\[DDG_PRELOAD\]\[KEY\]\s+"
            r"party=(\d+)\s+"
            r"bytes=(\d+)\s+"
            r"chunk_bytes=(\d+)\s+"
            r"runtime_us=(\d+)\s+"
            r"mode=([^\s]+)"
        )

        if args.full_key_ram:
            for party in (0, 1):
                eval_log = (
                    logs /
                    f"eval_p{party}.log"
                )

                eval_text = eval_log.read_text(
                    errors="ignore"
                )

                matches = preload_re.findall(
                    eval_text
                )

                if len(matches) != 1:
                    raise RuntimeError(
                        f"expected exactly one preload record "
                        f"for P{party}, got {len(matches)}"
                    )

                p_txt, bytes_txt, chunk_txt, us_txt, mode = (
                    matches[0]
                )

                if int(p_txt) != party:
                    raise RuntimeError(
                        f"preload party mismatch: "
                        f"expected={party}, actual={p_txt}"
                    )

                preload_info[party] = {
                    "bytes": int(bytes_txt),
                    "chunk_bytes": int(chunk_txt),
                    "runtime_us": int(us_txt),
                    "mode": mode,
                }

        # --------------------------------------------------------
        # Parse evaluator profiles.
        # --------------------------------------------------------
        eval_fields = (
            "input_load_us",
            "key_wait_us",
            "key_read_us",
            "h2d_us",
            "sync_us",
            "compute_us",
            "comm_bytes",
        )

        eval_totals = {
            party: parse_profile_totals(
                log=(
                    logs /
                    f"eval_p{party}.log"
                ),
                pattern=EVAL_PROFILE_RE,
                expected_party=party,
                expected_chunks=chunks,
                fields=eval_fields,
                label=f"evaluator P{party}",
            )
            for party in (0, 1)
        }

        p0_log = (
            logs /
            "eval_p0.log"
        )

        text = p0_log.read_text(
            errors="ignore"
        )

        pairs = AFF_GLOBAL_RE.findall(
            text
        )

        out_map = {
            int(i): float(v)
            for i, v in pairs
        }

        expected_indices = set(
            range(padded_n)
        )

        if set(out_map) != expected_indices:
            raise RuntimeError(
                "unexpected padded affinity indices: "
                f"{sorted(out_map)}"
            )

        padded_outputs = [
            out_map[i]
            for i in range(padded_n)
        ]

        outputs = (
            padded_outputs[:N]
        )

        throughput = (
            N * 1_000_000.0 /
            online_wall_us
        )

        # --------------------------------------------------------
        # Reporting.
        # --------------------------------------------------------
        print()
        print(
            "========================================"
        )
        print(
            "MACHINE-READABLE TIMING"
        )
        print(
            "========================================"
        )

        print(
            "[DDG_TIME][SCHEMA] "
            "version=4 "
            "status=PRE_COMPLIANCE "
            "key_mode=" +
            (
                "offline_ram_direct"
                if args.full_key_ram
                else "offline_full_file"
            )
        )

        print(
            "[DDG_TIME][PREPROCESS_UNTIMED] "
            f"samples={N} "
            f"micro_batch={B} "
            f"chunks={chunks} "
            f"chunk_materialize_us={materialize_us}"
        )

        print(
            "[DDG_TIME][OFFLINE_DEALER] "
            f"runtime_us="
            f"{meta.get('dealer_total_wall_us')} "
            "included_in_online_wall=0"
        )

        print(
            "[DDG_TIME][OFFLINE_KEY_DISTRIBUTION] "
            f"bytes_party0={k0.stat().st_size} "
            f"bytes_party1={k1.stat().st_size} "
            "runtime_us=NA "
            "included_in_online_wall=0"
        )

        if args.full_key_ram:
            for party in (0, 1):
                pi = preload_info[party]

                print(
                    "[DDG_TIME][KEY_PRELOAD] "
                    f"party={party} "
                    f"bytes={pi['bytes']} "
                    f"chunk_bytes={pi['chunk_bytes']} "
                    f"runtime_us={pi['runtime_us']} "
                    f"mode={pi['mode']} "
                    "included_in_online_wall=0 "
                    "included_in_setup_wall=1"
                )

        print(
            "[DDG_TIME][SETUP_TOTAL] "
            "status=PRE_COMPLIANCE "
            f"runtime_us={setup_wall_us} "
            "included_in_online_compute=0 "
            "included_in_end_to_end=1"
        )

        print(
            "[DDG_TIME][PROTEIN] "
            "mode=timed_fp32_gpu "
            f"runtime_us={protein_runtime_us} "
            "included_in_online_wall=1"
        )

        for party in (0, 1):
            e = eval_totals[party]

            print(
                "[DDG_TIME][EVALUATOR] "
                f"party={party} "
                f"chunks={chunks} "
                f"input_load_us="
                f"{e['input_load_us']} "
                f"key_wait_us="
                f"{e['key_wait_us']} "
                f"key_read_us="
                f"{e['key_read_us']} "
                f"h2d_us={e['h2d_us']} "
                f"sync_us={e['sync_us']} "
                f"compute_us={e['compute_us']} "
                f"comm_bytes={e['comm_bytes']}"
            )

        print(
            "[DDG_TIME][COMPLIANCE] "
            "status=PRE_COMPLIANCE "
            "dealer=offline_reported_separately "
            "key_distribution=offline_reported_separately "
            "key_preload=" +
            (
                "direct_full_ram_setup_reported_separately "
                if args.full_key_ram
                else "sequential_file_reads_in_online_compute "
            ) +
            "public_protein_model=timed_fp32_gpu "
            "secret_adj_norm=" +
            (
                "secure_online"
                if secure_adj_norm
                else "legacy_precomputed"
            )
        )

        print(
            "[DDG_TIME][END_TO_END] "
            "status=PRE_COMPLIANCE "
            "scope=setup_plus_online "
            f"samples={N} "
            f"micro_batch={B} "
            f"chunks={chunks} "
            f"runtime_us={end_to_end_us} "
            f"throughput_samples_s="
            f"{N * 1_000_000.0 / end_to_end_us:.6f} "
            "includes_preprocess=0 "
            "includes_offline_dealer=0 "
            "includes_setup=1 "
            "includes_online_compute=1"
        )

        print(
            "[DDG_TIME][ONLINE_COMPUTE] "
            "status=PRE_COMPLIANCE "
            f"samples={N} "
            f"micro_batch={B} "
            f"chunks={chunks} "
            f"padded_samples={padded_n - N} "
            f"bw={args.bw} "
            f"scale={args.scale} "
            f"key_chunk_bytes="
            f"{key_chunk_bytes} "
            f"runtime_us="
            f"{online_wall_us} "
            f"throughput_samples_s="
            f"{throughput:.6f} "
            f"mpc_comm_bytes_party0="
            f"{eval_totals[0]['comm_bytes']} "
            f"mpc_comm_bytes_party1="
            f"{eval_totals[1]['comm_bytes']}"
        )

        print()
        print(
            "========================================"
        )
        print(
            "LOGICAL RESULTS"
        )
        print(
            "========================================"
        )

        for i, value in enumerate(
            outputs
        ):
            print(
                f"AFFINITY_GLOBAL[{i}]="
                f"{value:.6f}"
            )

        if padded_n != N:
            print(
                f"[driver] trimmed "
                f"{padded_n - N} padding output(s)"
            )

        print()
        print(
            f"PASS: N={N}, fixed B={B}, "
            f"chunks={chunks}, "
            f"padded={padded_n}, "
            f"returned={len(outputs)}"
        )

        print(
            "OFFLINE/ONLINE SEPARATION: PASS"
        )

        print(
            "Dealer process was never started "
            "by this online runner."
        )

        success = True
        return 0

    except Exception:
        for proc in (
            eval_procs.values()
        ):
            stop_process(
                proc
            )

        for fh in (
            eval_files.values()
        ):
            if not fh.closed:
                fh.close()

        raise

    finally:
        if success and not args.keep_work:
            shutil.rmtree(
                work_root,
                ignore_errors=True,
            )
        elif not success:
            print()
            print(
                "[driver] FAILURE: retaining "
                f"work root {work_root}"
            )
        else:
            print(
                f"[driver] keep work root: "
                f"{work_root}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
