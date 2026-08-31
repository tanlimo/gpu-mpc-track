"""
Dense graph construction for the MPC affinity path (spec §6).

Converts a SMILES string into the fixed-size dense representation that gets
secret-shared into MPC:

    X      : (nmax, FEAT_DIM)  node features, L1-normalised per real atom,
                               zero-padded for padding nodes.
    A_hat  : (nmax, nmax)      symmetric-normalised adjacency  D^{-1/2}(A+I)D^{-1/2},
                               computed in plaintext (spec §5) — padding rows/cols 0.
    mask   : (nmax,)           binary 1 for real atoms, 0 for padding (secret-shared).

This mirrors DeepDTAGen/create_data.py::smile_to_graph but produces the dense,
fixed-Nmax form required for GPU-MPC instead of the variable-size sparse form.
"""
import numpy as np
from rdkit import Chem

# ── atom featurisation (identical to DeepDTAGen/create_data.py) ───────────────

_ATOM_SYMBOLS = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
    'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
    'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
    'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown',
]
_DEGREE      = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_NUM_HS      = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_IMP_VALENCE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_FORMAL_CHG  = [-1, -2, 1, 2, 0]
_HYBRID      = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def _one_of_k(x, allowable):
    return [x == a for a in allowable]


def _one_of_k_unk(x, allowable):
    # Matches create_data.py::one_of_k_encoding_unk exactly, including the
    # trailing "[x not in allowable]" flag. After the reassignment x is always
    # in `allowable`, so that flag is always 0 — but it still adds one dim.
    if x not in allowable:
        x = allowable[-1]
    return [x == a for a in allowable] + [x not in allowable]


def atom_features(atom) -> np.ndarray:
    """Per-atom feature vector (before L1 normalisation)."""
    feats = (
        _one_of_k_unk(atom.GetSymbol(), _ATOM_SYMBOLS)
        + _one_of_k(atom.GetDegree(), _DEGREE)
        + _one_of_k_unk(atom.GetTotalNumHs(), _NUM_HS)
        + _one_of_k_unk(atom.GetImplicitValence(), _IMP_VALENCE)
        + _one_of_k_unk(atom.GetFormalCharge(), _FORMAL_CHG)
        + _one_of_k_unk(atom.GetHybridization(), _HYBRID)
        + [atom.GetIsAromatic()]
        + [atom.IsInRing()]
    )
    return np.array(feats, dtype=np.float32)


# feature dimension produced by atom_features.
# 5 fields use _one_of_k_unk which appends an extra flag element (always 0
# after x is remapped, but still contributes to the dimension):
#   symbol(44+1) + degree(11) + numHs(11+1) + implValence(11+1)
#   + formalChg(5+1) + hybrid(5+1) + isAromatic(1) + isInRing(1)  = 94
FEAT_DIM = (len(_ATOM_SYMBOLS) + 1) + len(_DEGREE) + (len(_NUM_HS) + 1) \
    + (len(_IMP_VALENCE) + 1) + (len(_FORMAL_CHG) + 1) + (len(_HYBRID) + 1) + 2


def count_atoms(smile: str) -> int:
    """Number of heavy atoms RDKit assigns to a SMILES (== graph node count)."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smile!r}")
    return mol.GetNumAtoms()



def smile_to_dense_raw_graph(smile: str, nmax: int = 138):
    """SMILES -> (X, A_raw, mask) for compliant MPC preprocessing.

    A_raw is the dense 0/1 molecular adjacency with self-loops added on
    real atoms, but WITHOUT degree normalization.

    This is the representation that may be secret-shared before MPC.
    D, D^{-1/2}, and A_norm must be derived from secret A_raw inside
    the timed secure inference path.

    Shapes:
        X      : (nmax, FEAT_DIM), float32
        A_raw  : (nmax, nmax),     float32 values in {0,1}
        mask   : (nmax,),           float32 values in {0,1}
    """
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(
            f"RDKit could not parse SMILES: {smile!r}"
        )

    c_size = mol.GetNumAtoms()
    if c_size > nmax:
        raise ValueError(
            f"molecule has {c_size} atoms > nmax={nmax}"
        )

    # Node features: identical to the legacy/reference path.
    X = np.zeros(
        (nmax, FEAT_DIM),
        dtype=np.float32,
    )

    for i, atom in enumerate(mol.GetAtoms()):
        f = atom_features(atom)
        X[i] = f / f.sum()

    # Unnormalised private adjacency.
    #
    # Self-loops are part of the graph representation here.
    # The forbidden precomputation is degree normalization:
    #
    #     D
    #     D^{-1/2}
    #     D^{-1/2} A D^{-1/2}
    #
    A_raw = np.zeros(
        (nmax, nmax),
        dtype=np.float32,
    )

    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        A_raw[a, b] = 1.0
        A_raw[b, a] = 1.0

    for i in range(c_size):
        A_raw[i, i] = 1.0

    mask = np.zeros(
        nmax,
        dtype=np.float32,
    )
    mask[:c_size] = 1.0

    return X, A_raw, mask


def smile_to_dense_graph(smile: str, nmax: int = 138):
    """SMILES → (X, A_hat, mask) dense fixed-size representation (spec §6)."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smile!r}")

    c_size = mol.GetNumAtoms()
    if c_size > nmax:
        raise ValueError(f"molecule has {c_size} atoms > nmax={nmax}")

    # 1. node features, L1-normalised per atom (matches create_data.py)
    X = np.zeros((nmax, FEAT_DIM), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        f = atom_features(atom)
        X[i] = f / f.sum()

    # 2. dense adjacency A + I over the real-atom block
    A = np.zeros((nmax, nmax), dtype=np.float32)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[a, b] = 1.0
        A[b, a] = 1.0
    for i in range(c_size):
        A[i, i] = 1.0  # self-loop

    # 3. symmetric normalisation Â = D^{-1/2}(A+I)D^{-1/2} on real block only
    A_hat = np.zeros((nmax, nmax), dtype=np.float32)
    sub = A[:c_size, :c_size]
    deg = sub.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    A_hat[:c_size, :c_size] = d_inv_sqrt[:, None] * sub * d_inv_sqrt[None, :]

    # 4. binary mask
    mask = np.zeros(nmax, dtype=np.float32)
    mask[:c_size] = 1.0

    return X, A_hat, mask
