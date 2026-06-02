"""
human_nontata_promoters — downstream classification task.

Dataset: Grešová et al. (2023), Genomic Benchmarks
  ~36k sequences, 251 bp each, binary (promoter vs non-promoter)

Label map:
  0 = non-promoter
  1 = promoter (non-TATA)

Downloads to /tmp/genomic_benchmarks to avoid NFS stale-handle issues
on shared servers where $HOME is a network mount.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from genomics_dcnn.data.preprocess import encode_sequences

LABEL_MAP = {0: "non-promoter", 1: "promoter"}
SEQ_LEN   = 251

CACHE_DIR = Path(os.environ.get("GENOMIC_CACHE_DIR", "/tmp/genomic_benchmarks"))
CLOUD_URL = (
    "https://storage.googleapis.com/genomic_benchmarks_cache/"
    "human_nontata_promoters_0.zip"
)


def _ensure_downloaded() -> Path:
    """
    Download and extract the dataset to /tmp if not already present.
    Using /tmp avoids NFS stale-file-handle errors on shared GPU servers.
    """
    dest = CACHE_DIR / "human_nontata_promoters" / "0"
    if dest.exists():
        return dest

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "human_nontata_promoters_0.zip"

    print(f"Downloading human_nontata_promoters → {CACHE_DIR} …")
    urllib.request.urlretrieve(CLOUD_URL, zip_path)

    print("Extracting …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(CACHE_DIR)
    zip_path.unlink()

    return dest


def _read_split(root: Path, split: str) -> tuple[list[str], list[int]]:
    """
    Read sequences and labels from the extracted directory structure:
      root/split/0/*.txt  — class 0
      root/split/1/*.txt  — class 1
    Each .txt file contains one DNA sequence string.
    """
    split_dir = root / split
    sequences, labels = [], []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = int(class_dir.name)
        for seq_file in sorted(class_dir.iterdir()):
            seq = seq_file.read_text().strip()
            if seq:
                sequences.append(seq)
                labels.append(label)
    return sequences, labels


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
    root      = _ensure_downloaded()
    sequences, labels = _read_split(root, split)

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
