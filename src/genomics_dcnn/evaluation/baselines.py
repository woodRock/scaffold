"""
Baseline classifiers for genomic sequence classification.

Feature representations:
  kmer_k3  — 4^3 = 64-dim normalised k-mer frequency vector
  kmer_k4  — 4^4 = 256-dim normalised k-mer frequency vector
  kmer_k6  — 4^6 = 4096-dim normalised k-mer frequency vector (sparse)
  onehot   — flattened one-hot [L × 4] (works for short sequences)

Classifiers: RF, SVM, MLP, Logistic Regression
Same CV protocol as GenomicsFinetuner: N seeds × k-fold → flat list of scores.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

NUC_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def kmer_features(sequences: list[str], k: int = 4) -> np.ndarray:
    """
    Compute normalised k-mer frequency vectors for a list of DNA strings.

    Returns [N, 4^k] float32 array. Unknown bases (N) are skipped.
    """
    vocab = {''.join(c): i for i, c in enumerate(product('ACGT', repeat=k))}
    V = len(vocab)
    X = np.zeros((len(sequences), V), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        total = 0
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j + k]
            if kmer in vocab:
                X[i, vocab[kmer]] += 1
                total += 1
        if total > 0:
            X[i] /= total
    return X


def onehot_features(X_onehot: np.ndarray) -> np.ndarray:
    """Flatten [N, L, 4] one-hot array → [N, L*4] float32."""
    return X_onehot.reshape(len(X_onehot), -1)


def _fit_model(model_name: str, X_tr: np.ndarray, y_tr: np.ndarray):
    if model_name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        return clf
    if model_name == "svm":
        pipe = Pipeline([("sc", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0))])
        pipe.fit(X_tr, y_tr)
        return pipe
    if model_name == "mlp":
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300,
                                   early_stopping=True, random_state=0)),
        ])
        pipe.fit(X_tr, y_tr)
        return pipe
    if model_name == "lr":
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=0)),
        ])
        pipe.fit(X_tr, y_tr)
        return pipe
    raise ValueError(f"Unknown model_name: {model_name}")


def baseline_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    seeds: list[int] | None = None,
    cv_folds: int = 5,
) -> dict[str, list[float]]:
    """
    Evaluate a baseline over N seeds × k-fold CV.

    model_name : 'rf' | 'svm' | 'mlp' | 'lr'

    Returns {'balanced_accuracy': [...], 'macro_f1': [...]}
    """
    if seeds is None:
        seeds = list(range(10))

    all_ba, all_f1 = [], []
    for seed in seeds:
        np.random.seed(seed)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        for tr_idx, te_idx in skf.split(X, y):
            clf   = _fit_model(model_name, X[tr_idx], y[tr_idx])
            preds = clf.predict(X[te_idx])
            all_ba.append(balanced_accuracy_score(y[te_idx], preds))
            all_f1.append(f1_score(y[te_idx], preds, average="macro", zero_division=0))

    return {"balanced_accuracy": all_ba, "macro_f1": all_f1}
