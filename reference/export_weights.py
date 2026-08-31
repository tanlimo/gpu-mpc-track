"""
Fixed-point weight export for the GPU-MPC backend.

Serialises the MPC-secured layers of an AffinityModel (GCN×3, Drug_FC×2,
fusion×4) into the raw binary format GPU-MPC's `GPUModel::initWeights` expects
(nn/orca/gpu_model.h): a header-less little-endian int64 blob, layers
concatenated in forward order, each layer emitting its weight matrix (row-major)
followed by its bias.

Weight matrices are transposed from PyTorch's (out, in) to (in, out) so the C++
matmul — which computes Y = X · W with X laid out (rows, in) — reads the kernel
in the (in, out) orientation sytorch/CUTLASS uses.

The GatedCNN protein encoder is intentionally NOT exported: protein sequences
are public, so P2 evaluates that path in plaintext and only secret-shares the
resulting 128-d Pvec at the fusion boundary.

A sidecar `<out>.json` manifest records per-layer name/shape/offset so the C++
loader and the tests agree on the byte layout.
"""
import json
import os

import numpy as np

BITWIDTH = 64
DEFAULT_SCALE = 12

# forward-order (attribute, count) of the MPC-secured layer groups
_MPC_GROUPS = ("gcn", "drug_fc", "fusion")


def _to_fixed_i64(x: np.ndarray, scale: int) -> np.ndarray:
    """round(x * 2**scale) as int64, matching to_fixed()."""
    return np.rint(np.asarray(x, dtype=np.float64) * (1 << scale)).astype(np.int64)


def dump_mpc_weights(model, out_path: str, scale: int = DEFAULT_SCALE) -> dict:
    """Serialise MPC-secured weights to `out_path` (+ `out_path`.json manifest).

    Returns the manifest dict.
    """
    chunks = []          # list of int64 arrays, concatenated in order
    layers_meta = []
    offset = 0           # element offset (int64 units)

    for group in _MPC_GROUPS:
        for i, (W, b) in enumerate(getattr(model, group)):
            Wt = np.ascontiguousarray(W.T)                 # (in, out)
            w_fixed = _to_fixed_i64(Wt.ravel(order="C"), scale)
            b_fixed = _to_fixed_i64(np.asarray(b).ravel(), scale)

            layers_meta.append({
                "name":     f"{group}.{i}",
                "W_shape":  [int(Wt.shape[0]), int(Wt.shape[1])],
                "b_shape":  [int(b_fixed.size)],
                "W_offset": offset,
                "b_offset": offset + w_fixed.size,
            })
            chunks.append(w_fixed)
            chunks.append(b_fixed)
            offset += w_fixed.size + b_fixed.size

    blob = np.concatenate(chunks).astype("<i8")
    blob.tofile(out_path)

    manifest = {
        "scale":          int(scale),
        "bitwidth":       BITWIDTH,
        "total_elements": int(blob.size),
        "layers":         layers_meta,
    }
    with open(out_path + ".json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_mpc_weights(out_path: str) -> dict:
    """Inverse of dump_mpc_weights — returns {name: (W_float, b_float)}.

    Reads the `<out_path>.json` manifest to recover shapes/offsets and
    dequantises each layer back to float64. W is returned in (in, out) layout
    (as stored); transpose to compare with the PyTorch (out, in) originals.
    """
    with open(out_path + ".json") as f:
        manifest = json.load(f)
    scale = manifest["scale"]
    raw = np.fromfile(out_path, dtype="<i8")

    out = {}
    for layer in manifest["layers"]:
        wo, bo = layer["W_offset"], layer["b_offset"]
        wsh = layer["W_shape"]
        w_n = wsh[0] * wsh[1]
        b_n = layer["b_shape"][0]
        W = raw[wo:wo + w_n].reshape(wsh).astype(np.float64) / (1 << scale)
        b = raw[bo:bo + b_n].astype(np.float64) / (1 << scale)
        out[layer["name"]] = (W, b)
    return out
