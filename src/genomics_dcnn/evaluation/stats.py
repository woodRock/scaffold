"""
Statistical comparison of classifier conditions.

Protocol: Mann-Whitney U test (two-sided) + Bonferroni correction
          + rank-biserial correlation as effect size.
Identical to chroma-dcnn/src/chroma_dcnn/evaluation/stats.py.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def rank_biserial_r(u_stat: float, n1: int, n2: int) -> float:
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def compare_conditions(
    results: dict[str, list[float]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pairwise Mann-Whitney U tests with Bonferroni correction."""
    condition_names = list(results.keys())
    pairs           = list(combinations(condition_names, 2))
    n_comparisons   = len(pairs)

    rows = []
    for a, b in pairs:
        vals_a = np.array(results[a])
        vals_b = np.array(results[b])
        u_stat, p_raw = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        p_corr = min(p_raw * n_comparisons, 1.0)
        r = rank_biserial_r(u_stat, len(vals_a), len(vals_b))
        rows.append({
            "condition_a": a,
            "condition_b": b,
            "mean_a": vals_a.mean(),
            "std_a":  vals_a.std(),
            "mean_b": vals_b.mean(),
            "std_b":  vals_b.std(),
            "U":            u_stat,
            "p_raw":        p_raw,
            "p_corrected":  p_corr,
            "effect_r":     r,
            "significant":  p_corr < alpha,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("p_corrected")


def print_results_table(
    all_results: dict[str, dict[str, list[float]]],
    metric: str = "balanced_accuracy",
) -> None:
    rows = []
    for cond, task_results in all_results.items():
        row = {"condition": cond}
        for task, scores in task_results.items():
            arr = np.array(scores)
            row[f"{task}_mean"] = arr.mean()
            row[f"{task}_std"]  = arr.std()
            row[f"{task}_n"]    = len(arr)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("condition")

    print(f"\n=== Results ({metric}) ===")
    for col in df.columns:
        if col.endswith("_mean"):
            task    = col.removesuffix("_mean")
            df[task] = df.apply(
                lambda r, c=col, t=task: f"{r[c]:.3f} ± {r[t+'_std']:.3f} (n={int(r[t+'_n'])})",
                axis=1,
            )
    display = [c for c in df.columns if not c.endswith(("_std", "_n"))]
    print(df[display].to_string())
