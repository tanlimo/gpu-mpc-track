"""
Additive secret-sharing of confidential drug inputs — the P1 (data owner) side.

Mirrors GPU-MPC's `writeSharesCpu` (experiments/*/share_data.cpp): each
plaintext value is quantised to the bw-bit ring at a fixed scale, then split
into two additive shares over Z_{2^bw}:

    share1 = random pad
    share2 = (fixed - pad)  mod 2^bw
    share1 + share2 ≡ fixed  (mod 2^bw)

  bw=64 (default): Z_{2^64} — production 64-bit backend, files are `<u8`.
  bw=32          : Z_{2^32} — 32-bit test ring, files are `<u4` (half bandwidth).

Only the confidential drug tensors are shared here — node features X, the
normalised adjacency A_hat, and the node mask. Protein sequences are public and
are NOT shared (P2 evaluates the GatedCNN on the cleartext protein and shares
only the resulting Pvec at the fusion boundary).

Shares are written as little-endian unsigned raw files, one pair per tensor:
`<out_dir>/{tensor}_share{party}.dat` with party ∈ {0,1}.
"""
import os
import numpy as np

from reference import mpc_config
from reference.dense_graph import smile_to_dense_graph

U64_MOD = 1 << 64
U32_MOD = 1 << 32
DEFAULT_SCALE = mpc_config.SCALE
DEFAULT_BW = mpc_config.BW

# Binary node mask is an arithmetic-shared bit tensor.
# Values are literal ring elements {0,1}, not fixed-point {0,2^scale}.
MASK_SCALE = 0


def _bw_dtype(bw: int):
    """Return (unsigned numpy dtype, signed numpy dtype, ring modulus) for bw."""
    if bw == 32:
        return np.uint32, np.int32, U32_MOD
    return np.uint64, np.int64, U64_MOD


def _to_ring(x_float: np.ndarray, scale: int, bw: int) -> np.ndarray:
    """Quantise to fixed-point then reinterpret as an unsigned bw-bit element."""
    udtype, sdtype, _ = _bw_dtype(bw)
    # quantise in int64 first (sufficient precision for both 32 and 64 bw)
    fixed = np.rint(np.asarray(x_float, dtype=np.float64) * (1 << scale)).astype(np.int64)
    # reduce into the signed bw-bit range, then bit-preserve to unsigned
    if bw < 64:
        half = np.int64(1) << np.int64(bw - 1)
        mask = (np.int64(1) << np.int64(bw)) - np.int64(1)
        fixed_mod = fixed & mask
        fixed = np.where(fixed_mod >= half, fixed_mod - (mask + np.int64(1)), fixed_mod)
    return fixed.astype(sdtype).view(udtype)


def split_shares(x_float, scale: int = DEFAULT_SCALE, seed: int = 0,
                 bw: int = DEFAULT_BW, mask_bw: int = None):
    """Split a float array into two additive shares over Z_{2^bw}.

    Returns (share1, share2) flat unsigned arrays (uint32 for bw=32, uint64
    for bw=64) such that share1 + share2 ≡ round(x*2^scale) (mod 2^bw).

    Args:
        mask_bw: If provided, generate pad in [0, 2^mask_bw) instead of [0, 2^bw).
                 Use mask_bw=14 for Orca small-mask scheme to avoid FSS truncation
                 precision loss from large masks wrapping around sign boundary.
    """
    _, _, modulus = _bw_dtype(bw)
    udtype = _bw_dtype(bw)[0]
    x = np.asarray(x_float, dtype=np.float64).ravel()
    fixed = _to_ring(x, scale, bw)

    rng = np.random.default_rng(seed)
    # ORCA SMALL-MASK: limit pad to mask_bw bits to avoid truncation errors
    pad_modulus = (1 << mask_bw) if mask_bw else modulus
    pad = rng.integers(0, pad_modulus, size=fixed.shape, dtype=udtype)
    share1 = pad
    share2 = (fixed - pad).astype(udtype)   # wraps mod 2^bw
    return share1, share2


def reconstruct(share1, share2, scale: int = DEFAULT_SCALE,
                bw: int = DEFAULT_BW) -> np.ndarray:
    """Inverse of split_shares — recombine shares to a float array."""
    _, sdtype, _ = _bw_dtype(bw)
    udtype = _bw_dtype(bw)[0]
    s1 = np.asarray(share1, dtype=udtype)
    s2 = np.asarray(share2, dtype=udtype)
    summed = (s1 + s2).astype(udtype)       # wraps mod 2^bw
    as_int = summed.view(sdtype)            # reinterpret as two's complement
    return as_int.astype(np.float64) / (1 << scale)


def _write_pair(x_float, out_dir: str, tensor: str, scale: int, seed: int,
                bw: int = DEFAULT_BW):
    s0, s1 = split_shares(x_float, scale=scale, seed=seed, bw=bw)
    nbytes = bw // 8
    dtype_str = f"<u{nbytes}"
    os.makedirs(out_dir, exist_ok=True)
    s0.astype(dtype_str).tofile(os.path.join(out_dir, mpc_config.share_filename(tensor, 0)))
    s1.astype(dtype_str).tofile(os.path.join(out_dir, mpc_config.share_filename(tensor, 1)))


def share_drug_graph(smile: str, out_dir: str,
                     scale: int = DEFAULT_SCALE, nmax: int = 138,
                     seed: int = 0, pool_dim: int = 376,
                     bw: int = DEFAULT_BW):
    """Secret-share the confidential drug graph of `smile` to share files.

    Writes 3 tensor pairs (x, adj, mask). Uses a distinct sub-seed per tensor so
    the random pads are independent.

    The node mask is emitted *pre-tiled* to (nmax, pool_dim): the GPU-MPC
    masked-max-pool kernel multiplies it element-wise against the post-GCN node
    embeddings H:(nmax, pool_dim), and sytorch's `_Mul` forbids broadcasting.
    Tiling is pure column replication, so tile-then-share is identical to
    share-then-tile — the mask stays secret (it reveals the atom count = the
    molecule size, which is P1-confidential). `pool_dim` must equal the final
    GCN width (376 for DeepDTAGen: 94→188→282→376).

    `bw` controls the ring width: 64 (default, production) writes `<u8` files;
    32 (test/debug) writes `<u4` files, halving the share bandwidth.
    """
    X, A_hat, mask = smile_to_dense_graph(smile, nmax)
    mask_tiled = np.broadcast_to(mask.reshape(nmax, 1),
                                 (nmax, pool_dim))          # (nmax, pool_dim)
    _write_pair(X,          out_dir, "x",    scale, seed + 0, bw=bw)
    _write_pair(A_hat,      out_dir, "adj",  scale, seed + 1, bw=bw)
    _write_pair(mask_tiled, out_dir, "mask", MASK_SCALE, seed + 2, bw=bw)
    return {"nmax": nmax, "scale": scale, "mask_scale": MASK_SCALE, "pool_dim": pool_dim, "bw": bw,
            "shapes": {"x": list(X.shape), "adj": list(A_hat.shape),
                       "mask": list(mask_tiled.shape)}}
