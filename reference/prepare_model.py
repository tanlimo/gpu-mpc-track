#!/usr/bin/env python3
"""Validate and prepare a DeepDTAGen checkpoint for MPC affinity inference.

Input:
    released / replacement DeepDTAGen .pth checkpoint

Output:
    weights.bin
        fixed-point MPC weights for:
          GCN x3
          Drug FC x2
          Fusion FC x4

    weights.bin.json
        binary layout manifest

    model_metadata.json
        validated checkpoint/model contract

The original .pth remains the public Protein GatedCNN checkpoint and is
passed to run_persistent_local.py via --protein-checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference import mpc_config
from reference.affinity_model import AffinityModel
from reference.export_weights import dump_mpc_weights


EXPECTED_SHAPES = {
    # Drug GCN
    "encoder.GraphConv1.bias": (188,),
    "encoder.GraphConv2.bias": (282,),
    "encoder.GraphConv3.bias": (376,),

    # Drug FC
    "encoder.Drug_FCs.0.weight": (1024, 376),
    "encoder.Drug_FCs.0.bias": (1024,),
    "encoder.Drug_FCs.3.weight": (128, 1024),
    "encoder.Drug_FCs.3.bias": (128,),

    # Fusion
    "fc.FC_layers.0.weight": (1024, 256),
    "fc.FC_layers.0.bias": (1024,),
    "fc.FC_layers.3.weight": (512, 1024),
    "fc.FC_layers.3.bias": (512,),
    "fc.FC_layers.6.weight": (256, 512),
    "fc.FC_layers.6.bias": (256,),
    "fc.FC_layers.9.weight": (1, 256),
    "fc.FC_layers.9.bias": (1,),

    # Public Protein GatedCNN
    "cnn.Protein_Embed.weight": (26, 128),

    "cnn.Protein_Conv1.weight": (32, 1000, 8),
    "cnn.Protein_Conv1.bias": (32,),
    "cnn.Protein_Gate1.weight": (32, 1000, 8),
    "cnn.Protein_Gate1.bias": (32,),

    "cnn.Protein_Conv2.weight": (64, 32, 8),
    "cnn.Protein_Conv2.bias": (64,),
    "cnn.Protein_Gate2.weight": (64, 32, 8),
    "cnn.Protein_Gate2.bias": (64,),

    "cnn.Protein_Conv3.weight": (96, 64, 8),
    "cnn.Protein_Conv3.bias": (96,),
    "cnn.Protein_Gate3.weight": (96, 64, 8),
    "cnn.Protein_Gate3.bias": (96,),

    "cnn.Protein_FC.weight": (128, 10272),
    "cnn.Protein_FC.bias": (128,),
}


def state_dict_from_checkpoint(obj):
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        return obj["state_dict"]

    if isinstance(obj, dict):
        return obj

    raise TypeError(
        "checkpoint must be a raw state_dict or contain a state_dict mapping"
    )


def find_gcn_weight(sd, prefix):
    candidates = [
        f"{prefix}.lin.weight",
        f"{prefix}.weight",
    ]

    for name in candidates:
        if name in sd:
            return name

    raise KeyError(
        f"missing GCN weight for {prefix}; tried {candidates}"
    )


def validate_checkpoint(sd):
    errors = []

    # --------------------------------------------------------
    # Fixed architecture tensors.
    # --------------------------------------------------------
    for name, expected in EXPECTED_SHAPES.items():
        if name not in sd:
            errors.append(
                f"missing tensor: {name}"
            )
            continue

        actual = tuple(sd[name].shape)

        if actual != expected:
            errors.append(
                f"{name}: expected shape={expected}, got={actual}"
            )

    # --------------------------------------------------------
    # GCN weights support both newer and older PyG naming.
    # Expected mathematical dimensions:
    #   94 -> 188
    #   188 -> 282
    #   282 -> 376
    #
    # Newer .lin.weight is usually (out,in).
    # Older form may be (in,out).
    # AffinityModel.from_pth already handles this distinction.
    # --------------------------------------------------------
    gcn_expected = [
        ("encoder.GraphConv1", 94, 188),
        ("encoder.GraphConv2", 188, 282),
        ("encoder.GraphConv3", 282, 376),
    ]

    gcn_keys = []

    for prefix, in_dim, out_dim in gcn_expected:
        try:
            key = find_gcn_weight(
                sd,
                prefix,
            )
        except KeyError as exc:
            errors.append(str(exc))
            continue

        shape = tuple(sd[key].shape)

        valid = {
            (out_dim, in_dim),
            (in_dim, out_dim),
        }

        if shape not in valid:
            errors.append(
                f"{key}: unexpected shape={shape}; "
                f"expected one of {sorted(valid)}"
            )

        gcn_keys.append(key)

    if errors:
        raise RuntimeError(
            "checkpoint architecture validation failed:\n  - "
            + "\n  - ".join(errors)
        )

    return gcn_keys


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="DeepDTAGen .pth checkpoint",
    )

    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="model preparation output directory",
    )

    ap.add_argument(
        "--scale",
        type=int,
        default=mpc_config.SCALE,
    )

    ap.add_argument(
        "--force",
        action="store_true",
    )

    args = ap.parse_args()

    checkpoint = (
        args.checkpoint
        .resolve()
    )

    output = (
        args.output
        .resolve()
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            checkpoint
        )

    if output.exists():
        if (
            not args.force
            and any(output.iterdir())
        ):
            raise FileExistsError(
                f"{output} is not empty; "
                "use --force to overwrite generated files"
            )
    else:
        output.mkdir(
            parents=True,
        )

    obj = torch.load(
        checkpoint,
        map_location="cpu",
    )

    sd = state_dict_from_checkpoint(
        obj
    )

    gcn_keys = validate_checkpoint(
        sd
    )

    # AffinityModel currently consumes the released raw state_dict
    # checkpoint format directly.
    if sd is not obj:
        raise RuntimeError(
            "wrapped state_dict checkpoint validated successfully, "
            "but AffinityModel.from_pth currently requires the raw "
            "released state_dict format"
        )

    model = AffinityModel.from_pth(
        checkpoint,
        device="cpu",
    )

    weights_path = (
        output /
        "weights.bin"
    )

    manifest_path = Path(
        str(weights_path) +
        ".json"
    )

    metadata_path = (
        output /
        "model_metadata.json"
    )

    if args.force:
        for p in (
            weights_path,
            manifest_path,
            metadata_path,
        ):
            p.unlink(
                missing_ok=True
            )

    manifest = dump_mpc_weights(
        model,
        str(weights_path),
        scale=args.scale,
    )

    metadata = {
        "schema_version": 1,

        "checkpoint":
            str(checkpoint),

        "checkpoint_sha256":
            sha256_file(checkpoint),

        "checkpoint_bytes":
            checkpoint.stat().st_size,

        "scale":
            int(args.scale),

        "bw":
            int(mpc_config.BW),

        "architecture":
            "DeepDTAGen affinity branch",

        "drug_private":
            True,

        "protein_public":
            True,

        "protein_execution":
            "timed_fp32_gpu_from_original_checkpoint",

        "mpc_weights_file":
            weights_path.name,

        "mpc_weights_manifest":
            manifest_path.name,

        "mpc_total_elements":
            manifest["total_elements"],

        "mpc_layers":
            [
                layer["name"]
                for layer in manifest["layers"]
            ],

        "gcn_checkpoint_keys":
            gcn_keys,

        "protein_checkpoint_prefix":
            "cnn.",

        "expected_dimensions": {
            "nmax": mpc_config.NMAX,
            "feat_dim": mpc_config.FEAT_DIM,
            "pool_dim": mpc_config.POOL_DIM,
            "protein_length": 1000,
            "protein_vector": 128,
            "drug_vector": 128,
            "fusion_input": 256,
        },
    }

    with metadata_path.open(
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    print("PREPARE MODEL: PASS")
    print(
        f"checkpoint = {checkpoint}"
    )
    print(
        f"weights    = {weights_path}"
    )
    print(
        f"scale      = {args.scale}"
    )
    print(
        f"MPC layers = {len(manifest['layers'])}"
    )
    print(
        "Protein    = original .pth cnn.* "
        "(timed FP32 GPU)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
