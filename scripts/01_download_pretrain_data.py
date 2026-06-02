"""
Download JASPAR2024 CORE vertebrates PWMs (pretraining library).

Analogous to scripts/01_download_pretrain_data.py in chroma-dcnn, which
fetches MoNA/MassBank EI-MS spectra. JASPAR PWMs are the "spectral library"
for DNA: known regulatory motifs the model learns to recognise unsupervised.

Usage:
  python scripts/01_download_pretrain_data.py
  python scripts/01_download_pretrain_data.py --out data/jaspar.json
"""

import argparse
from genomics_dcnn.data.download import download_jaspar


def main(out_path: str) -> None:
    download_jaspar(out_path)
    print("\nDone. Next step:")
    print("  python scripts/03_pretrain_genomics.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download JASPAR PWMs")
    ap.add_argument("--out", default="data/jaspar.json")
    args = ap.parse_args()
    main(args.out)
