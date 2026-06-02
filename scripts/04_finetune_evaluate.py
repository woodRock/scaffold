"""
Fine-tune and evaluate GenomicsCNN on human_nontata_promoters.

Usage:
  python scripts/04_finetune_evaluate.py
  python scripts/04_finetune_evaluate.py --device cuda:0
  python scripts/04_finetune_evaluate.py --label_fraction 0.05
  python scripts/04_finetune_evaluate.py --label_fraction 0.05 --device mps

--label_fraction controls what proportion of the *training* split is used in
each CV fold. The validation split is always the full held-out fold, so the
metric is comparable across fractions. Use values in (0, 1]:
  1.0   — full data (default)
  0.25  — 25% of training labels per fold
  0.05  — 5%  of training labels per fold
  0.01  — 1%  of training labels per fold

Results are written to results/promoters_results_lf{fraction}.json so that
runs at different fractions do not overwrite each other.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from genomics_dcnn.downstream.promoters import load_all_splits
from genomics_dcnn.evaluation.stats import compare_conditions, print_results_table
from genomics_dcnn.training.finetune_genomics import GenomicsFinetuner


def main(config_path: str, device: str | None, label_fraction: float) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = cfg["task"].get("cv_seeds", list(range(10)))

    print(f"Loading human_nontata_promoters … (label_fraction={label_fraction})")
    X, y = load_all_splits()

    finetuner = GenomicsFinetuner(cfg, X, y, device=device, label_fraction=label_fraction)
    results   = finetuner.run_all_conditions(seeds=seeds)

    ba_results    = {cond: metrics["balanced_accuracy"] for cond, metrics in results.items()}
    comparison_df = compare_conditions(ba_results)
    print("\n--- Pairwise comparisons (balanced_accuracy) ---")
    print(comparison_df.to_string(index=False))

    tag = f"lf{label_fraction:.2f}".replace(".", "p")   # e.g. lf0p05
    comparison_df.to_csv(output_dir / f"comparisons_promoters_{tag}.csv", index=False)
    print_results_table({cond: {"promoters": r["balanced_accuracy"]} for cond, r in results.items()})

    out_path = output_dir / f"promoters_results_{tag}.json"
    with open(out_path, "w") as f:
        json.dump({
            cond: {m: [float(v) for v in vals] for m, vals in r.items()}
            for cond, r in results.items()
        }, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate GenomicsCNN on human_nontata_promoters")
    ap.add_argument("--config",         default="configs/finetune_promoters.yaml")
    ap.add_argument("--device",         type=str,   default=None)
    ap.add_argument("--label_fraction", type=float, default=1.0,
                    help="Fraction of training labels to use per fold (default: 1.0 = full data)")
    args = ap.parse_args()

    if not (0.0 < args.label_fraction <= 1.0):
        ap.error("--label_fraction must be in (0, 1]")

    main(args.config, args.device, args.label_fraction)
