"""
human_nontata_promoters — downstream classification task.

Dataset: Grešová et al. (2023), Genomic Benchmarks
  ~36k sequences, 251 bp each, binary (promoter vs non-promoter)
  Available via the genomic_benchmarks pip package.

Label map:
  0 = non-promoter
  1 = promoter (non-TATA)
"""

from __future__ import annotations

import numpy as np

from genomics_dcnn.data.preprocess import encode_sequences

LABEL_MAP = {0: "non-promoter", 1: "promoter"}
SEQ_LEN   = 251


def load_human_nontata_promoters(
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and one-hot encode the human_nontata_promoters dataset.

    Parameters
    ----------
    split : 'train' or 'test'

    Returns
    -------
    X : [N, SEQ_LEN, 4]  float32 one-hot
    y : [N]              int64 labels (0 = non-promoter, 1 = promoter)
    """
    try:
        from genomic_benchmarks.dataset_getters.pytorch_datasets import (
            HumanNontataPromoters,
        )
    except ImportError as e:
        raise ImportError(
            "genomic_benchmarks is required. Install with:\n"
            "  pip install genomic-benchmarks"
        ) from e

    ds = HumanNontataPromoters(split=split, version=0)

    sequences, labels = [], []
    for seq, label in ds:
        sequences.append(str(seq))
        labels.append(int(label))

    X = encode_sequences(sequences, seq_len=SEQ_LEN)
    y = np.array(labels, dtype=np.int64)

    classes, counts = np.unique(y, return_counts=True)
    print(f"human_nontata_promoters [{split}]: {len(X)} sequences, L={SEQ_LEN}")
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({LABEL_MAP.get(c, '?')}): {n} sequences")

    return X, y


def load_all_splits() -> tuple[np.ndarray, np.ndarray]:
    """Load train + test splits concatenated (for cross-validated evaluation)."""
    X_tr, y_tr = load_human_nontata_promoters("train")
    X_te, y_te = load_human_nontata_promoters("test")
    X = np.concatenate([X_tr, X_te], axis=0)
    y = np.concatenate([y_tr, y_te], axis=0)
    print(f"Combined: {len(X)} sequences")
    return X, y
