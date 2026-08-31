"""Single source of truth for the DeepDTAGen MPC fixed-point + file-naming
contract. Both the Python offline pipeline and the C++/CUDA online binary must
agree on these values (see gpu_mpc/deepdtagen_inference.cu)."""

# Validated production arithmetic for the current persistent-v2 path.
# BW=32 remains supported for experiments, but is not the current
# correctness candidate.
BW = 64            # ring Z_{2^64}
SCALE = 12         # 12 fractional fixed-point bits
NMAX = 138         # padded graph nodes
FEAT_DIM = 94      # node feature width
POOL_DIM = 376     # final GCN width == pooled embedding width

PROTEIN_EMB_FILE = "protein_emb.dat"

def share_filename(tensor: str, party: int) -> str:
    """0-based, prefix-free name the C++ loader reads (deepdtagen_inference.cu)."""
    assert party in (0, 1)
    return f"{tensor}_share{party}.dat"

def key_filename(bw: int = BW, scale: int = SCALE) -> str:
    """Must equal the C++ expName: 'DeepDTAGen_' + bw + '_' + scale."""
    return f"DeepDTAGen_{bw}_{scale}"
