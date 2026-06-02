"""
Preprocessing utilities for genomic sequences.
"""

from __future__ import annotations

import numpy as np

from genomics_dcnn.data.datasets import one_hot_encode


def encode_sequences(
    sequences: list[str],
    seq_len: int | None = None,
) -> np.ndarray:
    """
    One-hot encode a list of DNA strings → [N, L, 4] float32 array.

    If seq_len is None, pads/truncates to the length of the first sequence.
    """
    L = seq_len if seq_len is not None else len(sequences[0])
    return np.stack([one_hot_encode(s, L) for s in sequences])
