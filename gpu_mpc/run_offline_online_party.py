#!/usr/bin/env python3
"""
Per-party online DeepDTAGen runner.

This launcher is designed for the current architecture:

    OFFLINE
        one logical trusted Dealer
          -> K0
          -> K1
          -> exits

    ONLINE
        Server P0                  Server P1
        ---------                  ---------
        share0 + K0                share1 + K1
        Evaluator P0               Evaluator P1
              |                         |
              +------ GPU-MPC ----------+
                     TCP 42003

A second TCP connection is used only for launcher coordination:

        P0 control <-------------> P1 control
                     TCP 42004

The control connection coordinates:
    * evaluator launch order
    * READY
    * public Protein execution
    * START release
    * completion status

It does NOT carry MPC secret shares or FSS key material.

Status:
    PRE_COMPLIANCE

The same script is used on both servers:

    --party 0   listening/server side
    --party 1   connecting/client side
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time


NMAX = 138
FEAT = 94
POOL = 376
PROTEIN_LEN = 1000

MPC_PORT = 42003
MPC_PORT_SECONDARY = 42006

AFF_GLOBAL_RE = re.compile(
    r"^AFFINITY_GLOBAL\[(\d+)\]=([-+0-9.eE]+)$",
    re.MULTILINE,
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open() as f:
        return json.load(f)


def stop_process(
    proc: subprocess.Popen | None,
) -> None:
    if proc is None:
        return

    if proc.poll() is not None:
        return

    proc.terminate()

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def print_log_tail(
    path: Path,
    lines: int = 100,
) -> None:
    if not path.is_file():
        print(f"[missing log] {path}")
        return

    rows = path.read_text(
        errors="ignore"
    ).splitlines()

    for row in rows[-lines:]:
        print(row)


def copy_range(
    src: Path,
    dst: Path,
    offset: int,
    length: int,
    buf_size: int = 8 * 1024 * 1024,
) -> None:
    with src.open("rb") as fi, dst.open("wb") as fo:
        fi.seek(offset)

        remaining = length

        while remaining:
            block = fi.read(
                min(buf_size, remaining)
            )

            if not block:
                raise RuntimeError(
                    f"unexpected EOF while slicing {src}"
                )

            fo.write(block)
            remaining -= len(block)


def source_strides(
    bw: int,
    party: int,
) -> dict[str, int]:
    elem = bw // 8

    layout = {
        f"x_share{party}.dat":
            NMAX * FEAT * elem,

        f"adj_share{party}.dat":
            NMAX * NMAX * elem,

        f"mask_share{party}.dat":
            NMAX * POOL * elem,
    }

    if party == 1:
        # target_ids.dat contains 1000 int64 indices/sample.
        layout["target_ids.dat"] = (
            PROTEIN_LEN * 8
        )

    return layout


def validate_source(
    source_dir: Path,
    party: int,
    bw: int,
    scale: int,
    requested_n: int,
) -> tuple[dict, dict[str, int], int]:
    meta = load_json(
        source_dir / "metadata.json"
    )

    if meta.get("bw") != bw:
        raise RuntimeError(
            "dataset metadata BW mismatch: "
            f"expected={bw}, actual={meta.get('bw')}"
        )

    if meta.get("scale") != scale:
        raise RuntimeError(
            "dataset metadata SCALE mismatch: "
            f"expected={scale}, "
            f"actual={meta.get('scale')}"
        )

    if meta.get("nmax") != NMAX:
        raise RuntimeError(
            "dataset NMAX mismatch"
        )

    if meta.get("feat_dim") != FEAT:
        raise RuntimeError(
            "dataset feature dimension mismatch"
        )

    if meta.get("pool_dim") != POOL:
        raise RuntimeError(
            "dataset pool dimension mismatch"
        )

    if meta.get("protein_public") is not True:
        raise RuntimeError(
            "dataset metadata must mark protein public"
        )

    if (
        meta.get("protein_model_output_precomputed")
        is not False
    ):
        raise RuntimeError(
            "production path must not precompute "
            "Protein model output"
        )

    if (
        meta.get("adj_semantics")
        != "raw_binary_adjacency_with_self_loops"
    ):
        raise RuntimeError(
            "dataset must contain raw binary adjacency "
            "with self-loops"
        )

    available = int(
        meta.get("num_samples", 0)
    )

    if requested_n > available:
        raise RuntimeError(
            f"requested N={requested_n}, "
            f"dataset contains {available}"
        )

    layout = source_strides(
        bw,
        party,
    )

    for name, stride in layout.items():
        path = source_dir / name

        if not path.is_file():
            raise FileNotFoundError(
                f"party {party} missing input: {path}"
            )

        expected = (
            available * stride
        )

        actual = path.stat().st_size

        if actual != expected:
            raise RuntimeError(
                f"{path}: expected {expected} bytes, "
                f"got {actual}"
            )

    # Privacy-isolation diagnostic only.
    peer = 1 - party

    peer_files = [
        source_dir / f"x_share{peer}.dat",
        source_dir / f"adj_share{peer}.dat",
        source_dir / f"mask_share{peer}.dat",
    ]

    if any(p.exists() for p in peer_files):
        print(
            "[DDG_WARN][ISOLATION] "
            f"party {party} source directory also contains "
            f"party {peer} private shares. "
            "This is acceptable for localhost launcher testing, "
            "but not for final two-server privacy isolation.",
            flush=True,
        )

    return meta, layout, available


def validate_party_key(
    key_dir: Path,
    party: int,
    samples: int,
    batch: int,
    chunks: int,
    bw: int,
    scale: int,
    secure_adj_norm: bool,
) -> tuple[dict, Path, int]:
    meta = load_json(
        key_dir / "metadata.json"
    )

    expected = {
        "mode": "offline_full_key_file",
        "logical_dealer_count": 1,
        "online_evaluator_started": False,
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
                f"offline metadata mismatch for {name}: "
                f"expected={value!r}, actual={actual!r}"
            )

    files = meta.get("key_files")

    if not isinstance(files, dict):
        raise RuntimeError(
            "offline metadata missing key_files"
        )

    key_name = files.get(
        f"party{party}"
    )

    if not key_name:
        raise RuntimeError(
            f"metadata missing party{party} key"
        )

    key_path = (
        key_dir /
        key_name
    )

    if not key_path.is_file():
        raise FileNotFoundError(
            f"missing K{party}: {key_path}"
        )

    expected_total = int(
        meta.get(
            "key_total_bytes_per_party",
            0,
        )
    )

    actual_total = (
        key_path.stat().st_size
    )

    if actual_total != expected_total:
        raise RuntimeError(
            f"K{party} byte mismatch: "
            f"expected={expected_total}, "
            f"actual={actual_total}"
        )

    if actual_total % chunks != 0:
        raise RuntimeError(
            "key size is not divisible by chunk count"
        )

    chunk_bytes = (
        actual_total //
        chunks
    )

    if (
        meta.get("key_chunk_bytes_per_party")
        != chunk_bytes
    ):
        raise RuntimeError(
            "key chunk byte mismatch"
        )

    peer = 1 - party
    peer_name = files.get(
        f"party{peer}"
    )

    if peer_name:
        peer_path = (
            key_dir /
            peer_name
        )

        if peer_path.exists():
            print(
                "[DDG_WARN][ISOLATION] "
                f"party {party} key directory also contains "
                f"K{peer}. "
                "This is acceptable for localhost launcher testing, "
                "but final Server P0/P1 directories must contain "
                "only their own party key.",
                flush=True,
            )

    return meta, key_path, chunk_bytes


def materialize_party_chunks(
    *,
    source_dir: Path,
    work_root: Path,
    layout: dict[str, int],
    samples: int,
    batch: int,
) -> int:
    chunks = (
        samples + batch - 1
    ) // batch

    offset = 0

    for chunk in range(chunks):
        real_batch = min(
            batch,
            samples - offset,
        )

        chunk_dir = (
            work_root /
            f"chunk_{chunk:05d}"
        )

        chunk_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        for name, stride in layout.items():
            src = (
                source_dir /
                name
            )

            dst = (
                chunk_dir /
                name
            )

            copy_range(
                src=src,
                dst=dst,
                offset=(
                    offset * stride
                ),
                length=(
                    real_batch * stride
                ),
            )

            # Zero padding for the final fixed-B chunk.
            with dst.open("r+b") as f:
                f.truncate(
                    batch * stride
                )

            expected = (
                batch * stride
            )

            if dst.stat().st_size != expected:
                raise RuntimeError(
                    f"{dst}: invalid padded size"
                )

        print(
            f"[party-input] chunk={chunk} "
            f"real={real_batch} "
            f"padded={batch - real_batch} "
            f"global={offset}.."
            f"{offset + real_batch - 1}",
            flush=True,
        )

        offset += real_batch

    return chunks


def make_eval_env(
    *,
    weights: Path,
    party: int,
    gpu: str,
    work_root: Path,
    chunks: int,
    ready_file: Path,
    start_file: Path,
    secure_adj_norm: bool,
    full_key_ram: bool,
) -> dict[str, str]:
    env = os.environ.copy()

    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    env["DDG_WEIGHTS_BIN"] = str(
        weights
    )

    if secure_adj_norm:
        env["DDG_SECURE_ADJ_NORM"] = "1"
    else:
        env.pop(
            "DDG_SECURE_ADJ_NORM",
            None,
        )

    env["DDG_EVAL_CHUNK_ROOT"] = str(
        work_root
    )

    env["DDG_EVAL_CHUNKS"] = str(
        chunks
    )

    env["DDG_EVAL_READY_FILE"] = str(
        ready_file
    )

    env["DDG_EVAL_START_FILE"] = str(
        start_file
    )

    if full_key_ram:
        env[
            "DDG_EVAL_FULL_KEY_RAM"
        ] = "1"
    else:
        env.pop(
            "DDG_EVAL_FULL_KEY_RAM",
            None,
        )

    # Explicitly disable old experimental paths.
    for name in (
        "DDG_SLACK_TRUNC",
        "DDG_LOCAL_TRUNC",
        "DDG_INFERENCE_ITERS",
        "DDG_DEALER_CHUNK_ROOT",
        "DDG_DEALER_CHUNKS",
        "DDG_DEALER_EXTERNAL_KEY_IO",
        "DDG_EVAL_EXTERNAL_KEY_IO",
        "DDG_EVAL_KEY_CHUNK_BYTES",
        "DDG_KEY_SLOT_ROOT",
        "DDG_KEYBUF_CAP_GB",
    ):
        env.pop(
            name,
            None,
        )

    return env


def launch_evaluator(
    *,
    binary: Path,
    weights: Path,
    key_dir: Path,
    work_root: Path,
    log_path: Path,
    party: int,
    peer_ip: str,
    batch: int,
    chunks: int,
    bw: int,
    scale: int,
    gpu: str,
    secure_adj_norm: bool,
    full_key_ram: bool,
    ready_file: Path,
    start_file: Path,
) -> tuple[subprocess.Popen, object]:
    chunk0 = (
        work_root /
        "chunk_00000"
    )

    env = make_eval_env(
        weights=weights,
        party=party,
        gpu=gpu,
        work_root=work_root,
        chunks=chunks,
        ready_file=ready_file,
        start_file=start_file,
        secure_adj_norm=secure_adj_norm,
        full_key_ram=full_key_ram,
    )

    key_arg = (
        str(key_dir) +
        "/"
    )

    cmd = [
        str(binary),
        str(bw),
        str(scale),
        "1",                  # evaluator role
        str(party),
        key_arg,
        str(chunk0),
        str(batch),
        peer_ip,
    ]

    fh = log_path.open("w")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return proc, fh


def wait_local_ready(
    *,
    ready_file: Path,
    proc: subprocess.Popen,
    log_path: Path,
    timeout: float,
    party: int,
) -> None:
    deadline = (
        time.monotonic() +
        timeout
    )

    while True:
        if ready_file.exists():
            print(
                f"[control] local P{party} READY",
                flush=True,
            )
            return

        rc = proc.poll()

        if rc is not None:
            print()
            print(
                f"===== EVALUATOR P{party} LOG TAIL ====="
            )
            print_log_tail(
                log_path
            )

            raise RuntimeError(
                f"Evaluator P{party} exited before READY: "
                f"rc={rc}"
            )

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timeout waiting for P{party} READY"
            )

        time.sleep(0.01)


def send_msg(
    stream,
    msg_type: str,
    **fields,
) -> None:
    obj = {
        "type": msg_type,
        **fields,
    }

    payload = (
        json.dumps(
            obj,
            separators=(",", ":"),
        ) +
        "\n"
    ).encode()

    stream.write(
        payload
    )

    stream.flush()


def recv_msg(
    stream,
    expected: str | None = None,
) -> dict:
    line = stream.readline()

    if not line:
        raise RuntimeError(
            "control connection closed unexpectedly"
        )

    obj = json.loads(
        line.decode()
    )

    if expected is not None:
        actual = obj.get("type")

        if actual != expected:
            raise RuntimeError(
                "control protocol mismatch: "
                f"expected={expected}, actual={actual}"
            )

    return obj


def open_control_server(
    *,
    bind_host: str,
    port: int,
    timeout: float,
):
    listener = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )

    listener.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1,
    )

    listener.bind(
        (bind_host, port)
    )

    listener.listen(1)
    listener.settimeout(
        timeout
    )

    print(
        f"[control] P0 listening on "
        f"{bind_host}:{port}",
        flush=True,
    )

    conn, addr = (
        listener.accept()
    )

    listener.close()

    conn.settimeout(
        None
    )

    print(
        f"[control] P1 connected from "
        f"{addr[0]}:{addr[1]}",
        flush=True,
    )

    return conn


def connect_control_client(
    *,
    host: str,
    port: int,
    timeout: float,
):
    deadline = (
        time.monotonic() +
        timeout
    )

    while True:
        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        try:
            sock.connect(
                (host, port)
            )

            sock.settimeout(
                None
            )

            print(
                f"[control] P1 connected to "
                f"{host}:{port}",
                flush=True,
            )

            return sock

        except OSError:
            sock.close()

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timeout connecting to "
                    f"control server {host}:{port}"
                )

            time.sleep(0.2)


def run_protein(
    *,
    repo: Path,
    checkpoint: Path,
    work_root: Path,
    chunks: int,
    batch: int,
    scale: int,
    bw: int,
    gpu: str,
) -> int:
    worker = (
        repo /
        "reference" /
        "run_protein_chunks.py"
    )

    if not worker.is_file():
        raise FileNotFoundError(
            worker
        )

    env = os.environ.copy()

    env[
        "CUDA_VISIBLE_DEVICES"
    ] = str(gpu)

    cmd = [
        sys.executable,
        str(worker),
        "--checkpoint",
        str(checkpoint),
        "--chunk-root",
        str(work_root),
        "--chunks",
        str(chunks),
        "--batch",
        str(batch),
        "--scale",
        str(scale),
        "--bw",
        str(bw),
    ]

    start_ns = (
        time.perf_counter_ns()
    )

    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
    )

    end_ns = (
        time.perf_counter_ns()
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "public Protein worker failed: "
            f"rc={proc.returncode}"
        )

    return (
        end_ns -
        start_ns
    ) // 1000


def parse_party0_outputs(
    *,
    log_path: Path,
    samples: int,
    padded_n: int,
) -> list[float]:
    text = log_path.read_text(
        errors="ignore"
    )

    pairs = (
        AFF_GLOBAL_RE.findall(
            text
        )
    )

    out_map = {
        int(i): float(v)
        for i, v in pairs
    }

    expected = set(
        range(padded_n)
    )

    if set(out_map) != expected:
        raise RuntimeError(
            "unexpected padded affinity indices: "
            f"expected={sorted(expected)}, "
            f"actual={sorted(out_map)}"
        )

    return [
        out_map[i]
        for i in range(samples)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run one DeepDTAGen online MPC party using "
            "pre-generated offline FSS keys."
        )
    )

    ap.add_argument(
        "source_dir",
        type=Path,
        help=(
            "prepared sample-major dataset directory; "
            "only this party's private share files are required"
        ),
    )

    ap.add_argument(
        "weights_bin",
        type=Path,
    )

    ap.add_argument(
        "offline_key_dir",
        type=Path,
        help=(
            "offline key directory; only this party's "
            "K0/K1 file is required"
        ),
    )

    ap.add_argument(
        "--party",
        type=int,
        choices=(0, 1),
        required=True,
    )

    ap.add_argument(
        "--peer-ip",
        required=True,
        help=(
            "Server P0 IP. "
            "P0 listens; P1 connects to this address."
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
        "--gpu",
        default="0",
    )

    ap.add_argument(
        "--protein-checkpoint",
        type=Path,
        default=None,
        help=(
            "required on party 1; public DeepDTAGen "
            "checkpoint containing cnn.*"
        ),
    )

    ap.add_argument(
        "--control-port",
        type=int,
        default=42004,
    )

    ap.add_argument(
        "--control-bind",
        default="0.0.0.0",
        help="P0 control-listener bind address",
    )

    ap.add_argument(
        "--peer-start-delay",
        type=float,
        default=0.0,
        help=(
            "optional delay between launching P0 and instructing "
            "P1 to launch; normally unnecessary because the "
            "underlying Peer client retries connections"
        ),
    )

    ap.add_argument(
        "--timeout",
        type=float,
        default=600.0,
    )

    ap.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help=(
            "large writable directory for temporary fixed-B "
            "party chunks; defaults to offline_key_dir parent"
        ),
    )

    ap.add_argument(
        "--full-key-ram",
        action="store_true",
    )

    ap.add_argument(
        "--legacy-precomputed-adj",
        action="store_true",
    )

    ap.add_argument(
        "--keep-work",
        action="store_true",
    )

    args = ap.parse_args()

    if args.num_samples <= 0:
        ap.error(
            "--num-samples must be > 0"
        )

    if not (
        1 <= args.micro_batch <= 128
    ):
        ap.error(
            "--micro-batch must be in [1,128]"
        )

    if args.bw not in (
        32,
        64,
    ):
        ap.error(
            "--bw must be 32 or 64"
        )

    if args.peer_start_delay < 0:
        ap.error(
            "--peer-start-delay must be >= 0"
        )

    party = args.party

    source_dir = (
        args.source_dir.resolve()
    )

    weights = (
        args.weights_bin.resolve()
    )

    key_dir = (
        args.offline_key_dir.resolve()
    )

    if not source_dir.is_dir():
        raise FileNotFoundError(
            source_dir
        )

    if not weights.is_file():
        raise FileNotFoundError(
            weights
        )

    if not key_dir.is_dir():
        raise FileNotFoundError(
            key_dir
        )

    if party == 1:
        if (
            args.protein_checkpoint
            is None
        ):
            ap.error(
                "--protein-checkpoint is required "
                "for --party 1"
            )

        protein_checkpoint = (
            args.protein_checkpoint.resolve()
        )

        if not protein_checkpoint.is_file():
            raise FileNotFoundError(
                protein_checkpoint
            )

    else:
        protein_checkpoint = None

    N = args.num_samples
    B = args.micro_batch

    chunks = (
        N + B - 1
    ) // B

    padded_n = (
        chunks * B
    )

    secure_adj_norm = (
        not args.legacy_precomputed_adj
    )

    dataset_meta, layout, available = (
        validate_source(
            source_dir=source_dir,
            party=party,
            bw=args.bw,
            scale=args.scale,
            requested_n=N,
        )
    )

    key_meta, party_key, key_chunk_bytes = (
        validate_party_key(
            key_dir=key_dir,
            party=party,
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

    if not binary.is_file():
        raise FileNotFoundError(
            binary
        )

    work_parent = (
        args.work_root.resolve()
        if args.work_root is not None
        else key_dir.parent.resolve()
    )

    work_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    work_root = Path(
        tempfile.mkdtemp(
            prefix=(
                f"ddg_party{party}_"
            ),
            dir=str(work_parent),
        )
    )

    logs = (
        work_root /
        "logs"
    )

    logs.mkdir()

    ready_file = (
        work_root /
        f"eval_p{party}.ready"
    )

    start_file = (
        work_root /
        f"eval_p{party}.start"
    )

    log_path = (
        logs /
        f"eval_p{party}.log"
    )

    success = False

    evaluator = None
    evaluator_fh = None
    control_sock = None
    control_stream = None

    print(
        "========================================"
    )
    print(
        "DeepDTAGen Per-Party Online Runner"
    )
    print(
        "========================================"
    )
    print(
        f"party           = {party}"
    )
    print(
        f"source          = {source_dir}"
    )
    print(
        f"weights         = {weights}"
    )
    print(
        f"party key       = {party_key}"
    )
    print(
        f"logical N       = {N}"
    )
    print(
        f"micro-batch B   = {B}"
    )
    print(
        f"chunks          = {chunks}"
    )
    print(
        f"padded N        = {padded_n}"
    )
    print(
        f"BW / SCALE      = "
        f"{args.bw} / {args.scale}"
    )
    print(
        f"GPU             = {args.gpu}"
    )
    print(
        f"MPC peer        = "
        f"{args.peer_ip}:{MPC_PORT},"
        f"{MPC_PORT_SECONDARY}"
    )
    print(
        f"control port    = "
        f"{args.control_port}"
    )
    print(
        "Dealer          = NOT STARTED"
    )
    print(
        "A_norm          = " +
        (
            "secure online"
            if secure_adj_norm
            else "legacy precomputed"
        )
    )
    print(
        "key mode        = " +
        (
            "offline full file -> direct full RAM"
            if args.full_key_ram
            else "offline full file -> sequential reads"
        )
    )
    print(
        "status          = PRE_COMPLIANCE"
    )
    print(
        f"work root       = {work_root}"
    )
    print()

    try:
        materialize_start_ns = (
            time.perf_counter_ns()
        )

        actual_chunks = (
            materialize_party_chunks(
                source_dir=source_dir,
                work_root=work_root,
                layout=layout,
                samples=N,
                batch=B,
            )
        )

        if actual_chunks != chunks:
            raise RuntimeError(
                "internal chunk count mismatch"
            )

        materialize_us = (
            time.perf_counter_ns() -
            materialize_start_ns
        ) // 1000

        config = {
            "samples": N,
            "micro_batch": B,
            "chunks": chunks,
            "bw": args.bw,
            "scale": args.scale,
            "secure_adj_norm":
                secure_adj_norm,
            "full_key_ram":
                bool(args.full_key_ram),
            "key_chunk_bytes":
                key_chunk_bytes,
        }

        if party == 0:
            control_sock = (
                open_control_server(
                    bind_host=
                        args.control_bind,
                    port=
                        args.control_port,
                    timeout=
                        args.timeout,
                )
            )

            control_stream = (
                control_sock.makefile(
                    "rwb"
                )
            )

            hello = recv_msg(
                control_stream,
                "HELLO",
            )

            remote_config = (
                hello.get("config")
            )

            if remote_config != config:
                raise RuntimeError(
                    "P0/P1 configuration mismatch:\n"
                    f"P0={config}\n"
                    f"P1={remote_config}"
                )

            send_msg(
                control_stream,
                "HELLO_ACK",
                config=config,
            )

            setup_start_ns = (
                time.perf_counter_ns()
            )

            print(
                "[control] launch Evaluator P0",
                flush=True,
            )

            evaluator, evaluator_fh = (
                launch_evaluator(
                    binary=binary,
                    weights=weights,
                    key_dir=key_dir,
                    work_root=work_root,
                    log_path=log_path,
                    party=0,
                    peer_ip=args.peer_ip,
                    batch=B,
                    chunks=chunks,
                    bw=args.bw,
                    scale=args.scale,
                    gpu=args.gpu,
                    secure_adj_norm=
                        secure_adj_norm,
                    full_key_ram=
                        args.full_key_ram,
                    ready_file=
                        ready_file,
                    start_file=
                        start_file,
                )
            )

            if args.peer_start_delay:
                print(
                    "[control] temporary "
                    "P0->P1 launch delay: "
                    f"{args.peer_start_delay:.3f} s",
                    flush=True,
                )

                time.sleep(
                    args.peer_start_delay
                )

            send_msg(
                control_stream,
                "LAUNCH",
            )

            wait_local_ready(
                ready_file=ready_file,
                proc=evaluator,
                log_path=log_path,
                timeout=args.timeout,
                party=0,
            )

            recv_msg(
                control_stream,
                "READY",
            )

            setup_end_ns = (
                time.perf_counter_ns()
            )

            setup_us = (
                setup_end_ns -
                setup_start_ns
            ) // 1000

            print(
                "[control] BOTH PARTIES READY",
                flush=True,
            )

            online_start_ns = (
                time.perf_counter_ns()
            )

            send_msg(
                control_stream,
                "RUN_PROTEIN",
            )

            protein_done = recv_msg(
                control_stream,
                "PROTEIN_DONE",
            )

            protein_us = int(
                protein_done[
                    "runtime_us"
                ]
            )

            print(
                "[control] remote Protein complete: "
                f"{protein_us} us",
                flush=True,
            )

            # Tell P1 to release its local START barrier.
            send_msg(
                control_stream,
                "RELEASE",
            )

            # Release local P0 START barrier.
            start_file.write_text(
                "1\n"
            )

            print(
                "[control] START released",
                flush=True,
            )

            local_rc = (
                evaluator.wait()
            )

            remote_done = recv_msg(
                control_stream,
                "EVAL_DONE",
            )

            remote_rc = int(
                remote_done["rc"]
            )

            online_end_ns = (
                time.perf_counter_ns()
            )

            online_us = (
                online_end_ns -
                online_start_ns
            ) // 1000

            end_to_end_us = (
                online_end_ns -
                setup_start_ns
            ) // 1000

            evaluator_fh.close()
            evaluator_fh = None

            if (
                local_rc != 0
                or remote_rc != 0
            ):
                print()
                print(
                    "===== P0 EVALUATOR LOG TAIL ====="
                )
                print_log_tail(
                    log_path
                )

                raise RuntimeError(
                    "distributed evaluator failure: "
                    f"P0={local_rc}, "
                    f"P1={remote_rc}"
                )

            outputs = (
                parse_party0_outputs(
                    log_path=log_path,
                    samples=N,
                    padded_n=padded_n,
                )
            )

            print()
            print(
                "========================================"
            )
            print(
                "DISTRIBUTED TIMING"
            )
            print(
                "========================================"
            )

            print(
                "[DDG_TIME][DISTRIBUTED_SCHEMA] "
                "version=1 "
                "status=PRE_COMPLIANCE "
                "launcher=per_party"
            )

            print(
                "[DDG_TIME][PREPROCESS_UNTIMED] "
                f"party=0 "
                f"runtime_us={materialize_us}"
            )

            print(
                "[DDG_TIME][DISTRIBUTED_SETUP] "
                f"runtime_us={setup_us} "
                "includes_eval_launch=1 "
                "includes_peer_setup=1 "
                "includes_key_preload=1"
            )

            print(
                "[DDG_TIME][PROTEIN] "
                "party=1 "
                "mode=timed_fp32_gpu "
                f"runtime_us={protein_us}"
            )

            print(
                "[DDG_TIME][DISTRIBUTED_ONLINE] "
                f"samples={N} "
                f"micro_batch={B} "
                f"chunks={chunks} "
                f"runtime_us={online_us} "
                f"throughput_samples_s="
                f"{N * 1_000_000.0 / online_us:.6f}"
            )

            print(
                "[DDG_TIME][DISTRIBUTED_END_TO_END] "
                f"runtime_us={end_to_end_us} "
                "scope=setup_plus_online "
                "includes_offline_dealer=0 "
                "status=PRE_COMPLIANCE"
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
                    f"{padded_n - N} "
                    "padding output(s)"
                )

            print()
            print(
                f"PASS: distributed per-party runner "
                f"N={N}, B={B}, chunks={chunks}, "
                f"returned={len(outputs)}"
            )

            print(
                "OFFLINE/ONLINE SEPARATION: PASS"
            )

            print(
                "Dealer process was never started."
            )

            success = True

        else:
            control_sock = (
                connect_control_client(
                    host=args.peer_ip,
                    port=args.control_port,
                    timeout=args.timeout,
                )
            )

            control_stream = (
                control_sock.makefile(
                    "rwb"
                )
            )

            send_msg(
                control_stream,
                "HELLO",
                party=1,
                config=config,
            )

            recv_msg(
                control_stream,
                "HELLO_ACK",
            )

            recv_msg(
                control_stream,
                "LAUNCH",
            )

            setup_start_ns = (
                time.perf_counter_ns()
            )

            print(
                "[control] launch Evaluator P1",
                flush=True,
            )

            evaluator, evaluator_fh = (
                launch_evaluator(
                    binary=binary,
                    weights=weights,
                    key_dir=key_dir,
                    work_root=work_root,
                    log_path=log_path,
                    party=1,
                    peer_ip=args.peer_ip,
                    batch=B,
                    chunks=chunks,
                    bw=args.bw,
                    scale=args.scale,
                    gpu=args.gpu,
                    secure_adj_norm=
                        secure_adj_norm,
                    full_key_ram=
                        args.full_key_ram,
                    ready_file=
                        ready_file,
                    start_file=
                        start_file,
                )
            )

            wait_local_ready(
                ready_file=ready_file,
                proc=evaluator,
                log_path=log_path,
                timeout=args.timeout,
                party=1,
            )

            send_msg(
                control_stream,
                "READY",
            )

            recv_msg(
                control_stream,
                "RUN_PROTEIN",
            )

            online_start_ns = (
                time.perf_counter_ns()
            )

            assert (
                protein_checkpoint
                is not None
            )

            print(
                "[control] run public Protein GatedCNN",
                flush=True,
            )

            protein_us = (
                run_protein(
                    repo=repo,
                    checkpoint=
                        protein_checkpoint,
                    work_root=
                        work_root,
                    chunks=chunks,
                    batch=B,
                    scale=args.scale,
                    bw=args.bw,
                    gpu=args.gpu,
                )
            )

            send_msg(
                control_stream,
                "PROTEIN_DONE",
                runtime_us=
                    protein_us,
            )

            recv_msg(
                control_stream,
                "RELEASE",
            )

            start_file.write_text(
                "1\n"
            )

            print(
                "[control] local START released",
                flush=True,
            )

            rc = (
                evaluator.wait()
            )

            online_end_ns = (
                time.perf_counter_ns()
            )

            local_online_us = (
                online_end_ns -
                online_start_ns
            ) // 1000

            evaluator_fh.close()
            evaluator_fh = None

            send_msg(
                control_stream,
                "EVAL_DONE",
                rc=rc,
                online_runtime_us=
                    local_online_us,
            )

            if rc != 0:
                print()
                print(
                    "===== P1 EVALUATOR LOG TAIL ====="
                )
                print_log_tail(
                    log_path
                )

                raise RuntimeError(
                    f"Evaluator P1 failed: rc={rc}"
                )

            print()
            print(
                "[DDG_TIME][PARTY1_ONLINE] "
                "status=PRE_COMPLIANCE "
                f"runtime_us={local_online_us}"
            )

            print(
                "PARTY 1: PASS"
            )

            success = True

        return 0

    except Exception:
        stop_process(
            evaluator
        )

        raise

    finally:
        if (
            evaluator_fh is not None
            and not evaluator_fh.closed
        ):
            evaluator_fh.close()

        if control_stream is not None:
            try:
                control_stream.close()
            except Exception:
                pass

        if control_sock is not None:
            try:
                control_sock.close()
            except Exception:
                pass

        if (
            success
            and not args.keep_work
        ):
            shutil.rmtree(
                work_root,
                ignore_errors=True,
            )

        elif not success:
            print(
                f"[driver] FAILURE: "
                f"retaining work root {work_root}"
            )

        else:
            print(
                f"[driver] keep work root: "
                f"{work_root}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
