"""
PyTorch Dataset classes for pretraining and downstream fine-tuning.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

NUC_MAP   = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
VOCAB_SIZE = 4


def one_hot_encode(seq: str, seq_len: int | None = None) -> np.ndarray:
    """
    Encode a DNA string to a float32 one-hot array of shape [L, 4].
    'N' / unknown bases get uniform 0.25 across all four channels.
    """
    L   = seq_len if seq_len is not None else len(seq)
    out = np.zeros((L, VOCAB_SIZE), dtype=np.float32)
    for i, nuc in enumerate(seq[:L]):
        idx = NUC_MAP.get(nuc.upper(), -1)
        if idx >= 0:
            out[i, idx] = 1.0
        else:
            out[i] = 0.25
    return out


class DNADataset(Dataset):
    """
    Labelled one-hot DNA dataset for fine-tuning.

    Returns
    -------
    x : [L, vocab_size]  float32
    y : int64 scalar
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.X[idx], self.y[idx]


class SyntheticDNADataset(Dataset):
    """
    Synthetic DNA sequences generated on-the-fly from JASPAR PWMs.

    Each sample: random uniform-ACGT background with K randomly chosen
    JASPAR motifs injected at random positions. Used for pretraining without
    touching any labelled benchmark data — no label leakage into CV folds.

    Analogy to chroma-dcnn:
      JASPAR PWMs     ↔  MoNA/MassBank EI-MS spectra (spectral library)
      motif injection ↔  compound Gaussian elution profiles

    Returns
    -------
    x : [seq_len, vocab_size]  float32 one-hot
    """

    def __init__(
        self,
        pwms: list[np.ndarray],
        n_samples: int = 100_000,
        seq_len: int = 251,
        n_motifs_range: tuple[int, int] = (0, 5),
    ) -> None:
        super().__init__()
        self.pwms = pwms
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_motifs_range = n_motifs_range
        print(
            f"SyntheticDNADataset: {len(pwms)} JASPAR PWMs → "
            f"{n_samples} synthetic sequences, L={seq_len}"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tensor:
        rng = np.random.default_rng(idx)
        seq = self._generate(rng)
        # one-hot encode integer sequence
        out = np.eye(VOCAB_SIZE, dtype=np.float32)[seq]   # [L, 4]
        return torch.from_numpy(out)

    def _generate(self, rng: np.random.Generator) -> np.ndarray:
        seq = rng.integers(0, 4, size=self.seq_len)   # uniform ACGT background
        lo, hi = self.n_motifs_range
        K = int(rng.integers(lo, hi + 1))
        for _ in range(K):
            pwm = self.pwms[int(rng.integers(len(self.pwms)))]   # [4, motif_len]
            L_m = pwm.shape[1]
            if L_m >= self.seq_len:
                continue
            pos   = int(rng.integers(0, self.seq_len - L_m))
            probs = pwm / (pwm.sum(axis=0, keepdims=True) + 1e-8)   # [4, L_m]
            for j in range(L_m):
                seq[pos + j] = int(rng.choice(4, p=probs[:, j]))
        return seq
