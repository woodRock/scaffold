"""
Pretrain GenomicsMaskedPredictor via masked nucleotide prediction on synthetic DNA.

Synthetic sequences are generated on-the-fly from JASPAR PWMs so no labelled
benchmark data is used — no leakage into fine-tuning CV folds.

Usage:
  python scripts/03_pretrain_genomics.py
  python scripts/03_pretrain_genomics.py --device cuda:0
  python scripts/03_pretrain_genomics.py --iterations 500   # quick smoke test
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse

import torch
import yaml

from genomics_dcnn.data.download import load_jaspar
from genomics_dcnn.training.pretrain_genomics import pretrain_genomics


def _device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(config_path: str, iterations_override: int | None, device_str: str | None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if iterations_override is not None:
        cfg["pretraining"]["n_iterations"] = iterations_override

    pwms, _, _ = load_jaspar(cfg["data"]["jaspar_path"])
    device = _device(device_str)
    print(f"Device: {device}")
    pretrain_genomics(cfg, pwms, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pretrain GenomicsMaskedPredictor")
    ap.add_argument("--config",     default="configs/pretrain.yaml")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--device",     type=str, default=None)
    args = ap.parse_args()
    main(args.config, args.iterations, args.device)
