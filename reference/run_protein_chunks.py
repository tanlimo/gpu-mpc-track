#!/usr/bin/env python3
"""Timed public Protein GatedCNN worker.

Loads the released cnn.* checkpoint once, keeps it on one GPU, then
materializes protein_emb.dat for every fixed-B chunk.

The protein is public, so this is ordinary FP32 CUDA computation, not MPC.
TF32 is disabled inside protein_runtime.py to preserve the released FP32
reference at the Q12 fusion boundary.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from protein_runtime import (
    load_protein_model,
    materialize_protein_emb,
)


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

    args = ap.parse_args()

    if args.chunks <= 0:
        ap.error("--chunks must be > 0")

    if args.batch <= 0:
        ap.error("--batch must be > 0")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    total_start_ns = time.perf_counter_ns()

    # --------------------------------------------------------
    # Model/CUDA initialization is deliberately part of this
    # worker and therefore part of the caller's timed wall.
    # --------------------------------------------------------
    load_start_ns = time.perf_counter_ns()

    model = load_protein_model(
        args.checkpoint,
        device="cuda:0",
    )

    torch.cuda.synchronize()

    load_end_ns = time.perf_counter_ns()

    chunk_total_us = 0

    for chunk in range(args.chunks):
        chunk_dir = (
            args.chunk_root /
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
            raise FileNotFoundError(target_path)

        start_ns = time.perf_counter_ns()

        materialize_protein_emb(
            model=model,
            target_ids_path=target_path,
            output_path=output_path,
            batch=args.batch,
            scale=args.scale,
            bw=args.bw,
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
            f"batch={args.batch} "
            f"runtime_us={elapsed_us}",
            flush=True,
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
