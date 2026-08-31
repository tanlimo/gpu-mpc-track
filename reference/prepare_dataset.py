#!/usr/bin/env python3
"""Prepare a new DeepDTAGen affinity dataset for the MPC inference pipeline.

This program performs ONLY input-format conversion and secret splitting.

It deliberately does NOT execute any model computation:
  * no GCN
  * no adjacency normalization
  * no Protein GatedCNN
  * no affinity FC

Private drug input:
    compound_iso_smiles
        -> dense X
        -> raw binary adjacency + self-loops
        -> node mask
        -> two additive shares

Public protein input:
    target_sequence
        -> exact DeepDTAGen integer encoding [1000]
        -> target_ids.dat

The timed inference runner later computes:
    secret A -> A_norm
    public target_ids -> FP32 GatedCNN -> protein embedding
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import sys

# Allow both:
#   python3 -m reference.prepare_dataset
# and:
#   python3 reference/prepare_dataset.py
REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from reference import mpc_config
from reference.dense_graph import smile_to_dense_raw_graph
from reference.protein_runtime import (
    MAX_PROTEIN_LEN,
    PROTEIN_DICT,
    encode_protein_sequence,
)
from reference import share_data


SCHEMA_VERSION = 1


def _ring_dtype(bw: int) -> np.dtype:
    if bw == 64:
        return np.dtype("<u8")
    if bw == 32:
        return np.dtype("<u4")
    raise ValueError(f"unsupported bw={bw}")


def _split_csprng(
    values,
    *,
    scale: int,
    bw: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Information-theoretic additive split using OS random ring elements.

    Unlike the deterministic split helper used by regression fixtures, this
    production dataset path does not use a reproducible PRNG seed.
    """
    ring_dtype = _ring_dtype(bw)

    fixed = share_data._to_ring(
        np.asarray(values),
        scale=scale,
        bw=bw,
    ).reshape(-1)

    random_bytes = secrets.token_bytes(
        fixed.size * ring_dtype.itemsize
    )

    share0 = np.frombuffer(
        random_bytes,
        dtype=ring_dtype,
    ).copy()

    # Unsigned subtraction wraps modulo 2^bw.
    share1 = (
        fixed.astype(ring_dtype, copy=False)
        - share0
    ).astype(ring_dtype, copy=False)

    return share0, share1


def _split_debug(
    values,
    *,
    scale: int,
    bw: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic split for regression/debug only."""
    s0, s1 = share_data.split_shares(
        values,
        scale=scale,
        seed=seed,
        bw=bw,
    )

    dtype = _ring_dtype(bw)

    return (
        s0.astype(dtype, copy=False),
        s1.astype(dtype, copy=False),
    )


def _write_shares(
    handles,
    tensor: str,
    values,
    *,
    scale: int,
    bw: int,
    deterministic_seed: int | None,
    seed_offset: int,
) -> None:
    if deterministic_seed is None:
        s0, s1 = _split_csprng(
            values,
            scale=scale,
            bw=bw,
        )
    else:
        s0, s1 = _split_debug(
            values,
            scale=scale,
            bw=bw,
            seed=deterministic_seed + seed_offset,
        )

    s0.tofile(handles[f"{tensor}_share0.dat"])
    s1.tofile(handles[f"{tensor}_share1.dat"])


def _validate_columns(
    df: pd.DataFrame,
    smiles_col: str,
    protein_col: str,
) -> None:
    missing = [
        name
        for name in (smiles_col, protein_col)
        if name not in df.columns
    ]

    if missing:
        raise ValueError(
            "missing required CSV column(s): "
            + ", ".join(missing)
            + f"; available={list(df.columns)}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="input CSV",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="prepared sample-major output directory",
    )

    ap.add_argument(
        "--smiles-column",
        default="compound_iso_smiles",
    )
    ap.add_argument(
        "--protein-column",
        default="target_sequence",
    )
    ap.add_argument(
        "--affinity-column",
        default="affinity",
        help=(
            "optional label column; ignored when absent "
            "and never consumed by MPC inference"
        ),
    )

    ap.add_argument(
        "--bw",
        type=int,
        default=mpc_config.BW,
        choices=(32, 64),
    )
    ap.add_argument(
        "--scale",
        type=int,
        default=mpc_config.SCALE,
    )
    ap.add_argument(
        "--nmax",
        type=int,
        default=mpc_config.NMAX,
    )
    ap.add_argument(
        "--pool-dim",
        type=int,
        default=mpc_config.POOL_DIM,
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="debug only: prepare first N rows",
    )
    ap.add_argument(
        "--deterministic-seed",
        type=int,
        default=None,
        help=(
            "DEBUG/REGRESSION ONLY. Production default uses "
            "OS CSPRNG randomness for secret splitting."
        ),
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output directory",
    )

    args = ap.parse_args()

    csv_path = args.csv.resolve()
    output = args.output.resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    if args.limit is not None and args.limit <= 0:
        ap.error("--limit must be >0")

    df = pd.read_csv(csv_path)

    _validate_columns(
        df,
        args.smiles_column,
        args.protein_column,
    )

    if args.limit is not None:
        df = df.iloc[:args.limit].copy()

    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("input dataset is empty")

    if output.exists():
        if not args.force:
            raise FileExistsError(
                f"{output} already exists; use --force to replace it"
            )
        shutil.rmtree(output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = output.with_name(
        output.name + f".tmp.{os.getpid()}"
    )

    if tmp.exists():
        shutil.rmtree(tmp)

    tmp.mkdir()

    file_names = [
        "x_share0.dat",
        "x_share1.dat",
        "adj_share0.dat",
        "adj_share1.dat",
        "mask_share0.dat",
        "mask_share1.dat",
        "target_ids.dat",
    ]

    handles = {}

    labels = []

    try:
        for name in file_names:
            handles[name] = (
                tmp / name
            ).open("wb")

        N = len(df)

        for pos, row in df.iterrows():
            smile = str(
                row[args.smiles_column]
            )

            protein = str(
                row[args.protein_column]
            )

            bad_chars = sorted(
                set(protein) -
                set(PROTEIN_DICT)
            )

            if bad_chars:
                raise ValueError(
                    f"row {pos}: unsupported protein "
                    f"character(s)={bad_chars}"
                )

            try:
                X, A_raw, mask = (
                    smile_to_dense_raw_graph(
                        smile,
                        nmax=args.nmax,
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    f"row {pos}: failed to convert SMILES "
                    f"{smile[:120]!r}"
                ) from exc

            if X.shape != (
                args.nmax,
                mpc_config.FEAT_DIM,
            ):
                raise RuntimeError(
                    f"row {pos}: unexpected X shape "
                    f"{X.shape}"
                )

            if A_raw.shape != (
                args.nmax,
                args.nmax,
            ):
                raise RuntimeError(
                    f"row {pos}: unexpected adjacency "
                    f"shape {A_raw.shape}"
                )

            if mask.shape != (
                args.nmax,
            ):
                raise RuntimeError(
                    f"row {pos}: unexpected mask "
                    f"shape {mask.shape}"
                )

            # Production adjacency contract:
            # literal ring values {0,1}, NOT Q(scale).
            if not np.all(
                (A_raw == 0) |
                (A_raw == 1)
            ):
                raise RuntimeError(
                    f"row {pos}: raw adjacency "
                    "contains values outside {0,1}"
                )

            mask_tiled = np.broadcast_to(
                mask.reshape(
                    args.nmax,
                    1,
                ),
                (
                    args.nmax,
                    args.pool_dim,
                ),
            )

            # Unique deterministic seeds across sample/tensor
            # when the debug mode is explicitly requested.
            seed_base = pos * 3

            _write_shares(
                handles,
                "x",
                X,
                scale=args.scale,
                bw=args.bw,
                deterministic_seed=
                    args.deterministic_seed,
                seed_offset=seed_base + 0,
            )

            _write_shares(
                handles,
                "adj",
                A_raw,
                scale=0,
                bw=args.bw,
                deterministic_seed=
                    args.deterministic_seed,
                seed_offset=seed_base + 1,
            )

            _write_shares(
                handles,
                "mask",
                mask_tiled,
                scale=0,
                bw=args.bw,
                deterministic_seed=
                    args.deterministic_seed,
                seed_offset=seed_base + 2,
            )

            target_ids = (
                encode_protein_sequence(
                    protein
                )
                .astype("<i8", copy=False)
            )

            if target_ids.shape != (
                MAX_PROTEIN_LEN,
            ):
                raise RuntimeError(
                    f"row {pos}: invalid protein "
                    f"encoding shape={target_ids.shape}"
                )

            target_ids.tofile(
                handles["target_ids.dat"]
            )

            if (
                args.affinity_column
                in df.columns
            ):
                labels.append(
                    float(
                        row[
                            args.affinity_column
                        ]
                    )
                )

            if (
                pos == 0
                or (pos + 1) % 100 == 0
                or pos + 1 == N
            ):
                print(
                    f"[prepare_dataset] "
                    f"{pos + 1}/{N}",
                    flush=True,
                )

        for fh in handles.values():
            fh.close()

        handles.clear()

        elem = args.bw // 8

        bytes_per_sample = {
            "x_share0.dat":
                args.nmax *
                mpc_config.FEAT_DIM *
                elem,
            "x_share1.dat":
                args.nmax *
                mpc_config.FEAT_DIM *
                elem,

            "adj_share0.dat":
                args.nmax *
                args.nmax *
                elem,
            "adj_share1.dat":
                args.nmax *
                args.nmax *
                elem,

            "mask_share0.dat":
                args.nmax *
                args.pool_dim *
                elem,
            "mask_share1.dat":
                args.nmax *
                args.pool_dim *
                elem,

            "target_ids.dat":
                MAX_PROTEIN_LEN * 8,
        }

        for name, stride in (
            bytes_per_sample.items()
        ):
            actual = (
                tmp / name
            ).stat().st_size

            expected = (
                len(df) * stride
            )

            if actual != expected:
                raise RuntimeError(
                    f"{name}: expected "
                    f"{expected} bytes, "
                    f"got {actual}"
                )

        label_file = None

        if labels:
            if len(labels) != len(df):
                raise RuntimeError(
                    "partial affinity label column"
                )

            label_file = "affinity.npy"

            np.save(
                tmp / label_file,
                np.asarray(
                    labels,
                    dtype=np.float64,
                ),
            )

        metadata = {
            "schema_version":
                SCHEMA_VERSION,
            "num_samples":
                int(len(df)),

            "input_csv":
                str(csv_path),
            "smiles_column":
                args.smiles_column,
            "protein_column":
                args.protein_column,
            "affinity_column":
                (
                    args.affinity_column
                    if label_file is not None
                    else None
                ),

            "bw":
                int(args.bw),
            "scale":
                int(args.scale),

            "nmax":
                int(args.nmax),
            "feat_dim":
                int(mpc_config.FEAT_DIM),
            "pool_dim":
                int(args.pool_dim),

            "x_scale":
                int(args.scale),
            "adj_scale":
                0,
            "adj_semantics":
                "raw_binary_adjacency_with_self_loops",
            "mask_scale":
                0,

            "protein_public":
                True,
            "protein_length":
                MAX_PROTEIN_LEN,
            "protein_encoding":
                "DeepDTAGen seq_cat indices 0..25",
            "protein_model_output_precomputed":
                False,

            "sharing":
                (
                    "debug_deterministic_numpy"
                    if args.deterministic_seed
                    is not None
                    else "os_csprng_additive_ring"
                ),

            "files":
                bytes_per_sample,
            "label_file":
                label_file,
        }

        with (
            tmp / "metadata.json"
        ).open("w") as f:
            json.dump(
                metadata,
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")

        tmp.replace(output)

        print()
        print(
            "PREPARE DATASET: PASS"
        )
        print(
            f"samples = {len(df)}"
        )
        print(
            f"output  = {output}"
        )
        print(
            "protein = public target_ids.dat; "
            "NO protein_emb.dat"
        )
        print(
            "adj     = raw binary + self-loops, "
            "scale=0"
        )

        return 0

    except Exception:
        for fh in handles.values():
            try:
                fh.close()
            except Exception:
                pass

        if tmp.exists():
            shutil.rmtree(
                tmp,
                ignore_errors=True,
            )

        raise


if __name__ == "__main__":
    raise SystemExit(main())
