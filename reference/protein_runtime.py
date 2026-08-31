"""Public plaintext Protein GatedCNN runtime for DeepDTAGen affinity inference.

The protein sequence is PUBLIC.  Therefore the GatedCNN itself does not need
MPC protection and is evaluated in ordinary FP32 on GPU.

In the final inference pipeline this model computation must still execute
inside the measured runtime.  Only its final 128-d public vector is quantized
to the MPC fixed-point scale at the fusion boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn


MAX_PROTEIN_LEN = 1000
PROTEIN_VOCAB = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
PROTEIN_DICT = {
    ch: i + 1
    for i, ch in enumerate(PROTEIN_VOCAB)
}


def encode_protein_sequence(seq: str) -> np.ndarray:
    """Match the original DeepDTAGen seq_cat() preprocessing exactly.

    Output:
        int64 array of shape (1000,)

    0 is padding. Protein alphabet characters map to 1..25.
    """
    out = np.zeros(
        MAX_PROTEIN_LEN,
        dtype=np.int64,
    )

    for i, ch in enumerate(
        str(seq)[:MAX_PROTEIN_LEN]
    ):
        out[i] = PROTEIN_DICT[ch]

    return out


class ProteinGatedCNN(nn.Module):
    """Affinity-path GatedCNN copied structurally from DeepDTAGen."""

    def __init__(self) -> None:
        super().__init__()

        # Original DeepDTAGen hyperparameters:
        #
        # Protein_Features = 25
        # Embed_dim        = 128
        # Num_Filters      = 32
        # K_size           = 8
        # Final_dim        = 128

        self.Protein_Embed = nn.Embedding(
            26,
            128,
        )

        # NOTE:
        # The original model intentionally feeds
        #
        #   Embedding(target): [B, 1000, 128]
        #
        # directly into Conv1d. Therefore 1000 is the channel dimension.
        self.Protein_Conv1 = nn.Conv1d(
            in_channels=1000,
            out_channels=32,
            kernel_size=8,
        )

        self.Protein_Gate1 = nn.Conv1d(
            in_channels=1000,
            out_channels=32,
            kernel_size=8,
        )

        self.Protein_Conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=8,
        )

        self.Protein_Gate2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=8,
        )

        self.Protein_Conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=96,
            kernel_size=8,
        )

        self.Protein_Gate3 = nn.Conv1d(
            in_channels=64,
            out_channels=96,
            kernel_size=8,
        )

        self.relu = nn.ReLU()

        # Three kernel-8 convolutions:
        #
        # 128 -> 121 -> 114 -> 107
        #
        # giving [B, 96, 107].
        self.Protein_FC = nn.Linear(
            96 * 107,
            128,
        )

    def forward(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """target: [B,1000] integer protein indices."""

        x = self.Protein_Embed(target)
        # [B,1000,128]

        conv = self.Protein_Conv1(x)
        gate = torch.sigmoid(
            self.Protein_Gate1(x)
        )
        x = self.relu(conv * gate)

        conv = self.Protein_Conv2(x)
        gate = torch.sigmoid(
            self.Protein_Gate2(x)
        )
        x = self.relu(conv * gate)

        conv = self.Protein_Conv3(x)
        gate = torch.sigmoid(
            self.Protein_Gate3(x)
        )
        x = self.relu(conv * gate)

        x = x.reshape(
            x.shape[0],
            96 * 107,
        )

        return self.Protein_FC(x)


def _extract_state_dict(
    checkpoint,
) -> Mapping[str, torch.Tensor]:
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        return checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "checkpoint is not a state_dict-like mapping"
        )

    return checkpoint


def load_protein_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
) -> ProteinGatedCNN:
    """Load only cnn.* tensors from the released DeepDTAGen checkpoint.

    DeepDTAGen's released reference uses ordinary FP32 arithmetic.
    On Ampere/Hopper GPUs, allowing TF32 changes the final Q12 protein
    embedding by several LSBs, so explicitly disable TF32 here.
    """

    # Preserve reference FP32 semantics on Ampere/Hopper GPUs.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state = _extract_state_dict(
        checkpoint
    )

    cnn_state = {}

    prefix = "cnn."

    for key, value in state.items():
        if key.startswith(prefix):
            cnn_state[
                key[len(prefix):]
            ] = value

    model = ProteinGatedCNN()

    result = model.load_state_dict(
        cnn_state,
        strict=True,
    )

    if (
        result.missing_keys
        or result.unexpected_keys
    ):
        raise RuntimeError(
            "cnn state_dict mismatch: "
            f"missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )

    model.to(device)
    model.eval()

    return model


@torch.inference_mode()
def protein_embedding(
    model: ProteinGatedCNN,
    target_ids,
) -> torch.Tensor:
    """Run FP32 plaintext GatedCNN.

    Returns:
        FP32 CUDA/CPU tensor [B,128]
    """

    if not torch.is_tensor(target_ids):
        target_ids = torch.as_tensor(
            target_ids,
            dtype=torch.long,
        )

    if target_ids.ndim == 1:
        target_ids = target_ids.unsqueeze(0)

    if (
        target_ids.ndim != 2
        or target_ids.shape[1] != MAX_PROTEIN_LEN
    ):
        raise ValueError(
            "expected target_ids shape [B,1000], "
            f"got {tuple(target_ids.shape)}"
        )

    device = next(
        model.parameters()
    ).device

    target_ids = target_ids.to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    )

    return model(target_ids)


def quantize_q(
    values,
    scale: int = 12,
) -> np.ndarray:
    """Quantize public FP32 protein vector to signed fixed-point integers."""

    if torch.is_tensor(values):
        values = (
            values
            .detach()
            .cpu()
            .numpy()
        )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return np.rint(
        values * (1 << scale)
    ).astype(np.int64)


def materialize_protein_emb(
    model: ProteinGatedCNN,
    target_ids_path: str | Path,
    output_path: str | Path,
    batch: int,
    scale: int = 12,
    bw: int = 64,
) -> np.ndarray:
    """Generate one MPC chunk's public protein embedding.

    Input:
        target_ids_path:
            public int64 protein indices, shape [B,1000]

    Model computation:
        FP32 plaintext GatedCNN on the model's device.

    Output:
        output_path:
            ring-encoded Q(scale) protein vectors, shape [B,128],
            in the exact binary format expected by the existing C++ MPC path.

    Returns:
        signed fixed-point integer ndarray [B,128], useful for regression tests.
    """

    if batch <= 0:
        raise ValueError(
            f"batch must be >0, got {batch}"
        )

    if bw not in (32, 64):
        raise ValueError(
            f"unsupported bw={bw}"
        )

    target_ids_path = Path(target_ids_path)
    output_path = Path(output_path)

    target = np.fromfile(
        target_ids_path,
        dtype="<i8",
    )

    expected = (
        batch * MAX_PROTEIN_LEN
    )

    if target.size != expected:
        raise RuntimeError(
            f"{target_ids_path}: "
            f"expected {expected} int64 values "
            f"for B={batch}, got {target.size}"
        )

    target = target.reshape(
        batch,
        MAX_PROTEIN_LEN,
    )

    if (
        target.min(initial=0) < 0
        or target.max(initial=0) > 25
    ):
        raise RuntimeError(
            f"{target_ids_path}: protein index "
            "outside expected [0,25]"
        )

    # --------------------------------------------------------
    # Timed model computation.
    # --------------------------------------------------------
    out = protein_embedding(
        model,
        target,
    )

    if tuple(out.shape) != (
        batch,
        128,
    ):
        raise RuntimeError(
            "unexpected protein output shape: "
            f"{tuple(out.shape)}"
        )

    q = quantize_q(
        out,
        scale=scale,
    )

    # --------------------------------------------------------
    # Convert signed fixed-point values into the exact
    # two's-complement ring representation consumed by C++.
    # --------------------------------------------------------
    if bw == 64:
        signed = q.astype(
            "<i8",
            copy=False,
        )

        ring = signed.view("<u8")

    else:
        lo = -(1 << 31)
        hi = (1 << 31) - 1

        if (
            np.any(q < lo)
            or np.any(q > hi)
        ):
            raise OverflowError(
                "protein Q values overflow signed BW32"
            )

        signed = q.astype(
            "<i4",
        )

        ring = signed.view("<u4")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ring.tofile(
        output_path
    )

    expected_bytes = (
        batch *
        128 *
        (bw // 8)
    )

    actual_bytes = (
        output_path.stat().st_size
    )

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"{output_path}: expected "
            f"{expected_bytes} bytes, "
            f"got {actual_bytes}"
        )

    return q
