#!/usr/bin/env python3
"""Timed public Protein GatedCNN worker.

Legacy mode:
    process starts -> load model -> run all chunks -> exit

Ready/start mode:
    process starts
      -> import PyTorch / initialize CUDA runtime
      -> signal READY
      -> wait for START
      -> load released cnn.* checkpoint
      -> run all chunks
      -> exit

The second mode allows Python/PyTorch/CUDA process initialization to overlap
with evaluator setup while keeping checkpoint loading and Protein model
computation after the timed START boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from protein_runtime import (
    load_protein_model,
    materialize_protein_emb,
)


def wait_for_start(
    *,
    ready_file: Path,
    start_file: Path,
    timeout: float,
) -> None:
    """Initialize CUDA, signal READY, then wait for START."""

    ready_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ready_file.unlink(
        missing_ok=True,
    )

    # Force CUDA runtime/context initialization during worker setup.
    #
    # No DeepDTAGen checkpoint is loaded and no model computation is
    # performed here.
    setup_start_ns = time.perf_counter_ns()

    _ = torch.empty(
        1,
        device="cuda:0",
    )

    torch.cuda.synchronize()

    ready_file.write_text(
        "1\n"
    )

    setup_end_ns = time.perf_counter_ns()

    print(
        "[DDG_PROFILE][PROTEIN_WORKER_READY] "
        f"runtime_us="
        f"{(setup_end_ns - setup_start_ns) // 1000}",
        flush=True,
    )

    wait_start_ns = time.perf_counter_ns()

    deadline = (
        time.monotonic() +
        timeout
    )

    while not start_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timeout waiting for Protein START: "
                f"{start_file}"
            )

        time.sleep(0.001)

    wait_end_ns = time.perf_counter_ns()

    print(
        "[DDG_PROFILE][PROTEIN_WORKER_START] "
        f"wait_us="
        f"{(wait_end_ns - wait_start_ns) // 1000}",
        flush=True,
    )


def run_protein_chunks(
    *,
    model,
    chunk_root: Path,
    chunks: int,
    batch: int,
    scale: int,
    bw: int,
) -> int:
    chunk_total_us = 0

    for chunk in range(chunks):
        chunk_dir = (
            chunk_root /
            f"chunk_{chunk:05d}"
        )

        target_path = (
            chunk_dir /
            "target_ids.dat"
        )

        output_path = (
            chunk_dir /
            "protein_emb.dat"
        )

        if not target_path.is_file():
            raise FileNotFoundError(
                target_path
            )

        start_ns = time.perf_counter_ns()

        materialize_protein_emb(
            model=model,
            target_ids_path=target_path,
            output_path=output_path,
            batch=batch,
            scale=scale,
            bw=bw,
        )

        torch.cuda.synchronize()

        end_ns = time.perf_counter_ns()

        elapsed_us = (
            end_ns - start_ns
        ) // 1000

        chunk_total_us += elapsed_us

        print(
            "[DDG_PROFILE][PROTEIN_CHUNK] "
            f"chunk={chunk} "
            f"batch={batch} "
            f"runtime_us={elapsed_us}",
            flush=True,
        )

    return chunk_total_us



def run_persistent_worker(
    *,
    model,
    worker_dir: Path,
    scale: int,
    bw: int,
) -> int:
    """
    Persistent Protein worker v1.

    The model is already loaded.
    Wait for command.json and execute one request.
    """

    worker_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ready = worker_dir / "ready"
    command = worker_dir / "command.json"
    done = worker_dir / "done"

    ready.write_text("1\n")

    print(
        "[DDG_PROFILE][PROTEIN_PERSISTENT_READY]",
        flush=True,
    )

    while not command.exists():
        time.sleep(0.01)

    req = json.loads(
        command.read_text()
    )

    chunk_total_us = run_protein_chunks(
        model=model,
        chunk_root=Path(req["chunk_root"]),
        chunks=int(req["chunks"]),
        batch=int(req["batch"]),
        scale=scale,
        bw=bw,
    )

    done.write_text(
        str(chunk_total_us)
    )

    print(
        "[DDG_PROFILE][PROTEIN_PERSISTENT_DONE] "
        f"chunk_runtime_us={chunk_total_us}",
        flush=True,
    )

    return 0



def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--chunk-root",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--chunks",
        type=int,
        required=True,
    )
    ap.add_argument(
        "--batch",
        type=int,
        required=True,
    )
    ap.add_argument(
        "--scale",
        type=int,
        required=True,
    )
    ap.add_argument(
        "--bw",
        type=int,
        required=True,
    )

    # Optional persistent ready/start lifecycle.
    ap.add_argument(
        "--ready-file",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--start-file",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--wait-timeout",
        type=float,
        default=600.0,
    )

    ap.add_argument(
        "--persistent-worker",
        action="store_true",
        help="keep Protein model alive and wait for command",
    )

    ap.add_argument(
        "--worker-dir",
        type=Path,
        default=None,
        help="directory used for persistent worker control files",
    )

    args = ap.parse_args()

    if args.chunks <= 0:
        ap.error("--chunks must be > 0")

    if args.batch <= 0:
        ap.error("--batch must be > 0")

    if args.wait_timeout <= 0:
        ap.error("--wait-timeout must be > 0")

    if args.persistent_worker:
        if args.worker_dir is None:
            ap.error(
                "--worker-dir required with --persistent-worker"
            )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            args.checkpoint
        )

    lifecycle_enabled = (
        args.ready_file is not None
        or args.start_file is not None
    )

    if lifecycle_enabled:
        if (
            args.ready_file is None
            or args.start_file is None
        ):
            ap.error(
                "--ready-file and --start-file "
                "must be provided together"
            )

        wait_for_start(
            ready_file=args.ready_file,
            start_file=args.start_file,
            timeout=args.wait_timeout,
        )

    # --------------------------------------------------------
    # Everything below this point remains the timed Protein
    # model work when ready/start mode is used.
    # --------------------------------------------------------
    total_start_ns = time.perf_counter_ns()

    load_start_ns = time.perf_counter_ns()

    model = load_protein_model(
        args.checkpoint,
        device="cuda:0",
    )

    torch.cuda.synchronize()

    load_end_ns = time.perf_counter_ns()

    if args.persistent_worker:
        return run_persistent_worker(
            model=model,
            worker_dir=args.worker_dir,
            scale=args.scale,
            bw=args.bw,
        )

    chunk_total_us = run_protein_chunks(
        model=model,
        chunk_root=args.chunk_root,
        chunks=args.chunks,
        batch=args.batch,
        scale=args.scale,
        bw=args.bw,
    )


    total_end_ns = time.perf_counter_ns()

    print(
        "[DDG_PROFILE][PROTEIN] "
        f"chunks={args.chunks} "
        f"batch={args.batch} "
        f"model_load_us="
        f"{(load_end_ns - load_start_ns) // 1000} "
        f"chunk_runtime_us={chunk_total_us} "
        f"total_us="
        f"{(total_end_ns - total_start_ns) // 1000}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
