"""
Run k-mer frequency baselines on human_nontata_promoters.

Feature representations:
  kmer_k3  — 64-dim (4^3) k-mer frequency
  kmer_k4  — 256-dim (4^4) k-mer frequency
  kmer_k6  — 4096-dim (4^6) k-mer frequency
  onehot   — flattened one-hot [251 × 4 = 1004-dim]

Classifiers: RF, SVM, MLP, Logistic Regression

Usage:
  python scripts/05_run_baselines.py
  python scripts/05_run_baselines.py --config configs/finetune_promoters.yaml
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from genomics_dcnn.downstream.promoters import load_all_splits
from genomics_dcnn.evaluation.baselines import baseline_cv, kmer_features, onehot_features
from genomics_dcnn.evaluation.stats import compare_conditions


def _save_and_compare(results: dict, output_dir: Path, name: str) -> None:
    ba_results    = {k: v["balanced_accuracy"] for k, v in results.items()}
    comparison_df = compare_conditions(ba_results)
    print(f"\n--- Baseline comparisons ({name}) ---")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(output_dir / f"{name}_comparisons.csv", index=False)
    with open(output_dir / f"{name}_results.json", "w") as f:
        json.dump({
            k: {m: [float(v) for v in vals] for m, vals in metrics.items()}
            for k, metrics in results.items()
        }, f, indent=2)
    print(f"Saved → {output_dir}/{name}_results.json")


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds    = cfg["task"].get("cv_seeds", list(range(10)))
    cv_folds = cfg["task"].get("cv_folds", 5)

    print("Loading human_nontata_promoters …")
    X_onehot, y = load_all_splits()

    # Recover raw sequences for k-mer features from the one-hot array
    nuc_chars = ['A', 'C', 'G', 'T']
    raw_seqs  = [
        ''.join(nuc_chars[np.argmax(X_onehot[i, j])] for j in range(X_onehot.shape[1]))
        for i in range(len(X_onehot))
    ]

    representations = {
        "kmer_k3": kmer_features(raw_seqs, k=3),
        "kmer_k4": kmer_features(raw_seqs, k=4),
        "kmer_k6": kmer_features(raw_seqs, k=6),
        "onehot":  onehot_features(X_onehot),
    }

    for rep_name, X_rep in representations.items():
        print(f"\n=== {rep_name} (shape={X_rep.shape}) ===")
        rep_results = {}
        for model_name in ["rf", "svm", "mlp", "lr"]:
            print(f"  Running {model_name} …")
            rep_results[model_name] = baseline_cv(
                X_rep, y, model_name, seeds=seeds, cv_folds=cv_folds,
            )
            ba = rep_results[model_name]["balanced_accuracy"]
            print(f"    BA: {np.mean(ba):.3f} ± {np.std(ba):.3f}")
        _save_and_compare(rep_results, output_dir, f"baseline_{rep_name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run baselines on human_nontata_promoters")
    ap.add_argument("--config", default="configs/finetune_promoters.yaml")
    args = ap.parse_args()
    main(args.config)
