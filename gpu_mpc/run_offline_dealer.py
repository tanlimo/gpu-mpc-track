#!/usr/bin/env python3
"""
DeepDTAGen offline Dealer runner.

This script implements ONE logical semi-honest Dealer.

The existing C++ key generator exposes a party argument because the
Dealer must produce one party-specific FSS key stream for each online
evaluator:

    keygen party=0 -> K0
    keygen party=1 -> K1

These two passes are executed SEQUENTIALLY by this script and both
complete before any online evaluator is started.

This script performs OFFLINE preprocessing only.  It does not start
the two online MPC evaluators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from run_chunked_local import inspect_source
from run_persistent_local import make_fixed_chunk, base_env


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def run_keygen(
    *,
    binary: Path,
    weights: Path,
    chunk_root: Path,
    chunk0: Path,
    key_dir: Path,
    party: int,
    chunks: int,
    batch: int,
    bw: int,
    scale: int,
    gpu: str,
    keybuf_cap_gb: int,
    secure_adj_norm: bool,
    log: Path,
) -> int:
    """
    Generate the complete sequential FSS key file for one online party.

    This is one pass of a single logical Dealer, not a separate trust domain.
    """
    env = base_env(
        weights,
        secure_adj_norm=secure_adj_norm,
    )

    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    env["DDG_DEALER_CHUNK_ROOT"] = str(chunk_root)
    env["DDG_DEALER_CHUNKS"] = str(chunks)

    # Keygen still uses one reusable per-chunk GPU/host key buffer.
    env["DDG_KEYBUF_CAP_GB"] = str(keybuf_cap_gb)

    # IMPORTANT:
    # Sequential full-key mode.
    #
    # Do NOT enable:
    #   DDG_DEALER_EXTERNAL_KEY_IO
    #   DDG_KEY_SLOT_ROOT
    #
    # The C++ dealer therefore appends every chunk into one complete
    # party-specific key file.
    env.pop("DDG_DEALER_EXTERNAL_KEY_IO", None)
    env.pop("DDG_KEY_SLOT_ROOT", None)
    env.pop("DDG_LEGACY_SLOT_IO", None)

    key_dir_arg = str(key_dir.resolve()) + "/"

    cmd = [
        str(binary),
        str(bw),
        str(scale),
        "0",                    # role = Dealer
        str(party),             # generate K_party
        key_dir_arg,
        str(chunk0),
        str(batch),
    ]

    print(
        f"[offline-dealer] generate K{party}: "
        f"chunks={chunks} B={batch} GPU={gpu}",
        flush=True,
    )

    start_ns = time.perf_counter_ns()

    with log.open("w") as f:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    end_ns = time.perf_counter_ns()

    runtime_us = (end_ns - start_ns) // 1000

    if proc.returncode != 0:
        print()
        print(
            f"===== OFFLINE DEALER K{party} LOG TAIL ====="
        )

        lines = log.read_text(
            errors="ignore"
        ).splitlines()

        for line in lines[-100:]:
            print(line)

        raise RuntimeError(
            f"offline Dealer K{party} failed: "
            f"rc={proc.returncode}"
        )

    print(
        f"[offline-dealer] K{party} complete: "
        f"{runtime_us / 1e6:.6f} s",
        flush=True,
    )

    return runtime_us


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate complete party-specific DeepDTAGen FSS key "
            "files before online MPC execution."
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
        "output_dir",
        type=Path,
        help="output directory for complete offline FSS keys",
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
        "--gpu",
        default="0",
        help=(
            "GPU used by the logical offline Dealer. "
            "K0 and K1 are generated sequentially on this GPU."
        ),
    )

    ap.add_argument(
        "--keybuf-cap-gb",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--legacy-precomputed-adj",
        action="store_true",
        help=(
            "debug/regression only; default uses secure online "
            "A -> A_norm preprocessing material"
        ),
    )

    ap.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output directory",
    )

    ap.add_argument(
        "--keep-work",
        action="store_true",
        help="keep temporary fixed-B chunk directories",
    )

    args = ap.parse_args()

    src = args.source_dir.resolve()
    weights = args.weights_bin.resolve()
    out = args.output_dir.resolve()

    if args.num_samples <= 0:
        ap.error("--num-samples must be > 0")

    if not (1 <= args.micro_batch <= 128):
        ap.error("--micro-batch must be in [1,128]")

    if args.bw not in (32, 64):
        ap.error("--bw must be 32 or 64")

    if args.scale <= 0 or args.scale >= args.bw:
        ap.error("--scale must satisfy 0 < scale < bw")

    if args.keybuf_cap_gb <= 0:
        ap.error("--keybuf-cap-gb must be > 0")

    if not src.is_dir():
        raise FileNotFoundError(src)

    if not weights.is_file():
        raise FileNotFoundError(weights)

    # Dealer does NOT need protein_emb.dat.
    #
    # Protein is public and its GatedCNN remains in the timed online
    # model path.  Offline key generation uses zero protein mask.
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

    B = args.micro_batch
    N = args.num_samples

    chunks = (N + B - 1) // B
    padded_n = chunks * B

    repo = Path(__file__).resolve().parent.parent

    binary = (
        repo /
        "gpu_mpc" /
        "deepdtagen_inference"
    )

    if not binary.is_file():
        raise FileNotFoundError(
            f"missing binary: {binary}; "
            f"build BW={args.bw} first"
        )

    if out.exists():
        if not args.force:
            raise RuntimeError(
                f"output already exists: {out}; "
                "use --force to replace it"
            )

        shutil.rmtree(out)

    out.mkdir(parents=True)

    logs = out / "logs"
    logs.mkdir()

    work_root = Path(
        tempfile.mkdtemp(
            prefix="deepdtagen_offline_dealer_chunks_",
            dir="/tmp",
        )
    )

    success = False

    try:
        print("========================================")
        print("DeepDTAGen Offline Dealer")
        print("========================================")
        print(f"source          = {src}")
        print(f"weights         = {weights}")
        print(f"output          = {out}")
        print(f"logical N       = {N}")
        print(f"micro-batch B   = {B}")
        print(f"chunks          = {chunks}")
        print(f"padded N        = {padded_n}")
        print(f"BW / SCALE      = {args.bw} / {args.scale}")
        print(f"Dealer GPU      = {args.gpu}")
        print(
            "A_norm mode     = " +
            (
                "legacy precomputed"
                if args.legacy_precomputed_adj
                else "secure A -> A_norm key material"
            )
        )
        print()
        print(
            "Trust model     = one logical offline Dealer; "
            "K0 and K1 generated sequentially"
        )
        print(
            "Online MPC      = NOT started by this script"
        )
        print()

        # ------------------------------------------------------------
        # Input-format-only chunk materialization.
        # ------------------------------------------------------------
        chunk_start_ns = time.perf_counter_ns()

        offset = 0

        for chunk in range(chunks):
            real_batch = min(
                B,
                N - offset,
            )

            dst = (
                work_root /
                f"chunk_{chunk:05d}"
            )

            make_fixed_chunk(
                src=src,
                dst=dst,
                offset_samples=offset,
                real_batch=real_batch,
                fixed_batch=B,
                layout=layout,
            )

            offset += real_batch

        chunk_end_ns = time.perf_counter_ns()

        chunk_materialize_us = (
            chunk_end_ns -
            chunk_start_ns
        ) // 1000

        chunk0 = work_root / "chunk_00000"

        if not chunk0.is_dir():
            raise RuntimeError(
                f"missing first chunk: {chunk0}"
            )

        # ------------------------------------------------------------
        # ONE logical Dealer, two sequential party-key passes.
        #
        # Absolutely no evaluator is started here.
        # ------------------------------------------------------------
        dealer_total_start_ns = (
            time.perf_counter_ns()
        )

        party_runtime_us = {}

        for party in (0, 1):
            party_runtime_us[party] = run_keygen(
                binary=binary,
                weights=weights,
                chunk_root=work_root,
                chunk0=chunk0,
                key_dir=out,
                party=party,
                chunks=chunks,
                batch=B,
                bw=args.bw,
                scale=args.scale,
                gpu=str(args.gpu),
                keybuf_cap_gb=args.keybuf_cap_gb,
                secure_adj_norm=(
                    not args.legacy_precomputed_adj
                ),
                log=logs / f"dealer_keygen_k{party}.log",
            )

        dealer_total_end_ns = (
            time.perf_counter_ns()
        )

        dealer_total_us = (
            dealer_total_end_ns -
            dealer_total_start_ns
        ) // 1000

        # Native C++ sequential key filenames.
        key_files = {
            party: (
                out /
                (
                    f"DeepDTAGen_{args.bw}_{args.scale}"
                    f"_party{party}"
                    f"_inference_key{party}.dat"
                )
            )
            for party in (0, 1)
        }

        for party, path in key_files.items():
            if not path.is_file():
                raise RuntimeError(
                    f"missing complete K{party}: {path}"
                )

            if path.stat().st_size <= 0:
                raise RuntimeError(
                    f"empty complete K{party}: {path}"
                )

        size0 = key_files[0].stat().st_size
        size1 = key_files[1].stat().st_size

        if size0 != size1:
            raise RuntimeError(
                "party key file sizes differ: "
                f"K0={size0}, K1={size1}"
            )

        if size0 % chunks != 0:
            raise RuntimeError(
                "complete key file size is not divisible "
                f"by chunk count: bytes={size0}, chunks={chunks}"
            )

        key_chunk_bytes = size0 // chunks

        metadata = {
            "schema_version": 1,
            "mode": "offline_full_key_file",
            "status": "PRE_COMPLIANCE",
            "logical_dealer_count": 1,
            "party_key_generation": "sequential",
            "online_evaluator_started": False,
            "samples": N,
            "micro_batch": B,
            "chunks": chunks,
            "padded_samples": padded_n - N,
            "bw": args.bw,
            "scale": args.scale,
            "secure_adj_norm": (
                not args.legacy_precomputed_adj
            ),
            "key_chunk_bytes_per_party": key_chunk_bytes,
            "key_total_bytes_per_party": size0,
            "chunk_materialize_us": chunk_materialize_us,
            "dealer_keygen_pass_wall_us": {
                "party0": party_runtime_us[0],
                "party1": party_runtime_us[1],
            },
            "dealer_total_wall_us": dealer_total_us,
            "timing_policy": {
                "dealer": "offline_reported_separately",
                "key_distribution": "not_measured_in_this_local_test",
                "online": "not_started",
            },
            "key_files": {
                "party0": key_files[0].name,
                "party1": key_files[1].name,
            },
        }

        with (out / "metadata.json").open("w") as f:
            json.dump(
                metadata,
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")

        print()
        print("========================================")
        print("OFFLINE DEALER RESULT")
        print("========================================")

        print(
            "[DDG_TIME][OFFLINE_INPUT_FORMAT] "
            f"runtime_us={chunk_materialize_us}"
        )

        for party in (0, 1):
            print(
                "[DDG_TIME][OFFLINE_DEALER_KEYGEN] "
                f"key_party={party} "
                f"runtime_us={party_runtime_us[party]} "
                "timing_scope=offline_separate"
            )

        print(
            "[DDG_TIME][OFFLINE_DEALER_TOTAL] "
            f"runtime_us={dealer_total_us} "
            "timing_scope=offline_separate"
        )

        print(
            "[DDG_TIME][OFFLINE_KEY_DISTRIBUTION] "
            f"bytes_party0={size0} "
            f"bytes_party1={size1} "
            "runtime_us=NA "
            "timing_scope=offline_separate"
        )

        print()
        print(f"K0 = {key_files[0]}")
        print(f"K1 = {key_files[1]}")
        print(
            f"key bytes/chunk/party = "
            f"{key_chunk_bytes:,}"
        )
        print(
            f"key bytes/party       = "
            f"{size0:,}"
        )
        print(
            f"total K0+K1 bytes     = "
            f"{size0 + size1:,}"
        )
        print()

        print(
            "OFFLINE DEALER: PASS"
        )
        print(
            "Dealer lifecycle complete; "
            "no online evaluator was started."
        )

        success = True

    finally:
        if args.keep_work:
            print(
                f"[offline-dealer] keep work root: "
                f"{work_root}"
            )
        else:
            shutil.rmtree(
                work_root,
                ignore_errors=True,
            )

        if not success:
            print(
                f"[offline-dealer] FAILED; "
                f"logs retained in {logs}"
            )


if __name__ == "__main__":
    main()
