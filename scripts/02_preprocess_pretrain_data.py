"""
(Optional) inspect and filter downloaded JASPAR PWMs.

Usage:
  python scripts/02_preprocess_pretrain_data.py
  python scripts/02_preprocess_pretrain_data.py --jaspar data/jaspar.npz
"""

import argparse

import numpy as np

from genomics_dcnn.data.download import load_jaspar


def main(jaspar_path: str) -> None:
    pwms, names, matrix_ids = load_jaspar(jaspar_path)

    print(f"\n{'ID':<15} {'Name':<20} {'Motif len':>10} {'GC bias':>10}")
    print("-" * 60)
    for pwm, name, mid in zip(pwms[:20], names[:20], matrix_ids[:20]):
        gc = (pwm[1].sum() + pwm[2].sum()) / (pwm.sum() + 1e-8)
        print(f"{mid:<15} {name:<20} {pwm.shape[1]:>10d} {gc:>10.2f}")

    if not pwms:
        print("\nNo PWMs loaded — run scripts/01_download_pretrain_data.py first.")
        return
    lengths = [p.shape[1] for p in pwms]
    print(f"\nMotif length — min: {min(lengths)}  max: {max(lengths)}  mean: {np.mean(lengths):.1f}")
    print(f"Total PWMs: {len(pwms)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inspect JASPAR PWMs")
    ap.add_argument("--jaspar", default="data/jaspar.json")
    args = ap.parse_args()
    main(args.jaspar)
