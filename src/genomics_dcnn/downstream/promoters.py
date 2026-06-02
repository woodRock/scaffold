"""
human_nontata_promoters — downstream classification task.

Dataset: Grešová et al. (2023), Genomic Benchmarks
  ~36k sequences, 251 bp each, binary (promoter vs non-promoter)

Label map:
  0 = non-promoter
  1 = promoter (non-TATA)

Set GENOMIC_CACHE_DIR to a shared persistent path on cluster filesystems
to avoid re-downloading on every node and to avoid NFS home-dir issues:

  export GENOMIC_CACHE_DIR=/vol/ecrg-solar/woodj4/genomic_benchmarks
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from genomics_dcnn.data.preprocess import encode_sequences

LABEL_MAP = {0: "non-promoter", 1: "promoter"}
SEQ_LEN   = 251

# Redirect genomic_benchmarks cache before the package touches the filesystem.
# Defaults to ~/genomic_benchmarks; override with GENOMIC_CACHE_DIR env var.
_CACHE = Path(os.environ.get("GENOMIC_CACHE_DIR", str(Path.home() / "genomic_benchmarks")))
_CACHE.mkdir(parents=True, exist_ok=True)

# Monkey-patch the cache path so genomic_benchmarks uses our directory.
import genomic_benchmarks.loc2seq.loc2seq as _loc2seq  # noqa: E402
_loc2seq.get_dataset_path = lambda: _CACHE


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
    from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanNontataPromoters

    print(f"  cache → {_CACHE}", flush=True)
    ds = HumanNontataPromoters(split=split, version=0)

    sequences, labels = [], []
    for seq, label in ds:
        sequences.append(str(seq))
        labels.append(int(label))

    X = encode_sequences(sequences, seq_len=SEQ_LEN)
    y = np.array(labels, dtype=np.int64)

    classes, counts = np.unique(y, return_counts=True)
    print(f"human_nontata_promoters [{split}]: {len(X)} sequences, L={SEQ_LEN}", flush=True)
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({LABEL_MAP.get(c, '?')}): {n} sequences", flush=True)

    return X, y


def load_all_splits() -> tuple[np.ndarray, np.ndarray]:
    """Load train + test splits concatenated (for cross-validated evaluation)."""
    X_tr, y_tr = load_human_nontata_promoters("train")
    X_te, y_te = load_human_nontata_promoters("test")
    X = np.concatenate([X_tr, X_te], axis=0)
    y = np.concatenate([y_tr, y_te], axis=0)
    print(f"Combined: {len(X)} sequences", flush=True)
    return X, y
