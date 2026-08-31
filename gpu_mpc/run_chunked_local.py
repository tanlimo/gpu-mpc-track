#!/usr/bin/env python3
"""
Local arbitrary-N DeepDTAGen MPC driver.

Input:
  One sample-major contiguous share directory, containing:
    x_share{0,1}.dat
    adj_share{0,1}.dat
    mask_share{0,1}.dat
    protein_emb.dat

Execution:
  N samples
      -> chunks of at most --micro-batch samples
      -> fresh MPC keys for every chunk
      -> run_local_2pc.sh
      -> concatenate AFFINITY outputs in original sample order

This is a LOCAL development/correctness driver. It is not the final
two-server submission launcher.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


NMAX = 138
FEAT = 94
POOL = 376
PROTEIN = 128

# The underlying runner intentionally preserves the historical output format:
#   BATCH=1 : AFFINITY=<value>
#   BATCH>1 : AFFINITY[i]=<value>
# Accept both here and normalize the singleton form to local index 0.
AFF_RE = re.compile(
    r"^AFFINITY(?:\[(\d+)\])?=([-+0-9.eE]+)$"
)


def bytes_per_sample(bw: int) -> dict[str, int]:
    if bw not in (32, 64):
        raise ValueError(f"unsupported bw={bw}")

    elem = bw // 8

    return {
        "x_share0.dat": NMAX * FEAT * elem,
        "x_share1.dat": NMAX * FEAT * elem,

        "adj_share0.dat": NMAX * NMAX * elem,
        "adj_share1.dat": NMAX * NMAX * elem,

        "mask_share0.dat": NMAX * POOL * elem,
        "mask_share1.dat": NMAX * POOL * elem,

        "protein_emb.dat": PROTEIN * elem,
    }


def inspect_source(src: Path, bw: int) -> tuple[int, dict[str, int]]:
    layout = bytes_per_sample(bw)
    counts = []

    for name, stride in layout.items():
        p = src / name

        if not p.is_file():
            raise FileNotFoundError(f"missing input file: {p}")

        size = p.stat().st_size

        if size % stride != 0:
            raise RuntimeError(
                f"{p}: size={size} is not divisible by "
                f"bytes/sample={stride}"
            )

        counts.append(size // stride)

    if len(set(counts)) != 1:
        raise RuntimeError(
            f"inconsistent sample counts: {counts}"
        )

    return counts[0], layout


def copy_range(
    src: Path,
    dst: Path,
    offset: int,
    length: int,
    buf_size: int = 8 * 1024 * 1024,
) -> None:
    """Copy one byte range without loading the whole source file."""
    with src.open("rb") as fi, dst.open("wb") as fo:
        fi.seek(offset)

        remaining = length

        while remaining:
            block = fi.read(min(buf_size, remaining))

            if not block:
                raise RuntimeError(
                    f"unexpected EOF while slicing {src}"
                )

            fo.write(block)
            remaining -= len(block)


def make_chunk(
    src: Path,
    dst: Path,
    offset_samples: int,
    batch: int,
    layout: dict[str, int],
) -> None:
    dst.mkdir(parents=True, exist_ok=False)

    for name, stride in layout.items():
        byte_offset = offset_samples * stride
        byte_length = batch * stride

        copy_range(
            src / name,
            dst / name,
            byte_offset,
            byte_length,
        )


def run_chunk(
    runner: Path,
    chunk_dir: Path,
    key_dir: Path,
    weights: Path,
    batch: int,
    bw: int,
    scale: int,
    chunk_id: int,
) -> list[float]:

    env = os.environ.copy()

    env["BW"] = str(bw)
    env["SCALE"] = str(scale)
    env["BATCH"] = str(batch)

    # Keep the validated default arithmetic path.
    env.pop("DDG_SLACK_TRUNC", None)
    env.pop("DDG_LOCAL_TRUNC", None)

    cmd = [
        str(runner),
        str(chunk_dir),
        str(key_dir),
        str(weights),
    ]

    print(
        f"\n===== CHUNK {chunk_id}: "
        f"BATCH={batch} =====",
        flush=True,
    )

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None

    local_outputs: dict[int, float] = {}

    for raw in proc.stdout:
        line = raw.rstrip("\n")

        m = AFF_RE.match(line)

        if m:
            raw_idx = m.group(1)

            # run_local_2pc.sh preserves two historical output forms:
            #   BATCH=1 : AFFINITY=<value>
            #   BATCH>1 : AFFINITY[i]=<value>
            # Normalize the singleton form to local index 0.
            if raw_idx is None:
                if batch != 1:
                    raise RuntimeError(
                        f"chunk {chunk_id}: unindexed AFFINITY output "
                        f"is only valid for BATCH=1: {line}"
                    )
                idx = 0
            else:
                idx = int(raw_idx)

            value = float(m.group(2))
            local_outputs[idx] = value

        # Prefix child output so only the final global outputs below
        # remain machine-readable AFFINITY_GLOBAL lines.
        print(f"[chunk {chunk_id}] {line}")

    rc = proc.wait()

    if rc != 0:
        raise RuntimeError(
            f"chunk {chunk_id} failed with return code {rc}; "
            f"keys retained at {key_dir}"
        )

    expected = set(range(batch))
    actual = set(local_outputs)

    if actual != expected:
        raise RuntimeError(
            f"chunk {chunk_id}: expected local affinity indices "
            f"{sorted(expected)}, got {sorted(actual)}"
        )

    return [local_outputs[i] for i in range(batch)]


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "source_dir",
        type=Path,
        help="contiguous sample-major MPC input directory",
    )
    ap.add_argument(
        "weights_bin",
        type=Path,
    )
    ap.add_argument(
        "key_root",
        type=Path,
        help="large-disk directory for transient MPC keys",
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
        "--keep-keys",
        action="store_true",
    )
    ap.add_argument(
        "--keep-work",
        action="store_true",
    )

    args = ap.parse_args()

    src = args.source_dir.resolve()
    weights = args.weights_bin.resolve()
    key_root = args.key_root.resolve()

    if args.num_samples <= 0:
        ap.error("--num-samples must be > 0")

    if not (1 <= args.micro_batch <= 128):
        ap.error("--micro-batch must be in [1,128]")

    if not weights.is_file():
        raise FileNotFoundError(weights)

    repo = Path(__file__).resolve().parent.parent
    runner = repo / "gpu_mpc" / "run_local_2pc.sh"

    if not runner.is_file():
        raise FileNotFoundError(runner)

    available, layout = inspect_source(src, args.bw)

    if args.num_samples > available:
        raise RuntimeError(
            f"requested N={args.num_samples}, "
            f"but source contains only {available}"
        )

    key_root.mkdir(parents=True, exist_ok=True)

    work_root = Path(
        tempfile.mkdtemp(
            prefix="deepdtagen_chunked_",
            dir="/tmp",
        )
    )

    print("========================================")
    print("DeepDTAGen arbitrary-N local driver")
    print("========================================")
    print(f"source        = {src}")
    print(f"available N   = {available}")
    print(f"requested N   = {args.num_samples}")
    print(f"micro-batch   = {args.micro_batch}")
    print(f"BW / SCALE    = {args.bw} / {args.scale}")
    print(f"work root     = {work_root}")
    print(f"key root      = {key_root}")

    outputs: list[float] = []

    try:
        offset = 0
        chunk_id = 0

        while offset < args.num_samples:
            batch = min(
                args.micro_batch,
                args.num_samples - offset,
            )

            end = offset + batch

            chunk_dir = (
                work_root
                / f"chunk_{chunk_id:05d}_samples_{offset}_{end - 1}"
            )

            key_dir = (
                key_root
                / f"chunk_{chunk_id:05d}_b{batch}"
            )

            if key_dir.exists():
                shutil.rmtree(key_dir)

            print(
                f"\n[driver] chunk={chunk_id} "
                f"global_samples={offset}..{end - 1} "
                f"B={batch}",
                flush=True,
            )

            make_chunk(
                src,
                chunk_dir,
                offset,
                batch,
                layout,
            )

            try:
                local = run_chunk(
                    runner=runner,
                    chunk_dir=chunk_dir,
                    key_dir=key_dir,
                    weights=weights,
                    batch=batch,
                    bw=args.bw,
                    scale=args.scale,
                    chunk_id=chunk_id,
                )
            except Exception:
                print(
                    f"[driver] FAILED chunk {chunk_id}; "
                    f"retaining {chunk_dir} and {key_dir}",
                    file=sys.stderr,
                )
                raise

            outputs.extend(local)

            if not args.keep_keys:
                shutil.rmtree(
                    key_dir,
                    ignore_errors=True,
                )

            # Chunk input has already been consumed.
            if not args.keep_work:
                shutil.rmtree(
                    chunk_dir,
                    ignore_errors=True,
                )

            offset = end
            chunk_id += 1

        if len(outputs) != args.num_samples:
            raise RuntimeError(
                f"internal error: expected {args.num_samples} outputs, "
                f"got {len(outputs)}"
            )

        print()
        print("========================================")
        print("GLOBAL RESULTS")
        print("========================================")

        for i, value in enumerate(outputs):
            print(
                f"AFFINITY_GLOBAL[{i}]={value:.6f}"
            )

        print()
        print(
            f"PASS: produced {len(outputs)} "
            f"ordered affinity predictions"
        )

        return 0

    finally:
        if not args.keep_work:
            # Empty after successful per-chunk cleanup.
            # On failure a retained non-empty chunk prevents accidental loss.
            try:
                work_root.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
