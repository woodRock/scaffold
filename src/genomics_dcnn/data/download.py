"""
Download JASPAR2024 CORE vertebrates PWMs from the flat-file bulk download.

Fetches a single .txt file (JASPAR PFM format) instead of paginating the REST
API — the list endpoint returns pfm=null and fetching 2000+ detail pages is slow.

Saves a JSON file containing:
  pwms       : list of [4, L_i] lists (position frequency matrices as ACGT rows)
  names      : list of TF names (str)
  matrix_ids : list of JASPAR matrix IDs (str)

JASPAR PFM format (one matrix):
  >MA0001.1 AGL3
  A  [ 0  3 79 ...]
  C  [94 75  4 ...]
  G  [ 1  0  3 ...]
  T  [ 2 19 11 ...]
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import requests

JASPAR_FLAT = (
    "https://jaspar.elixir.no/download/data/2024/CORE/"
    "JASPAR2024_CORE_vertebrates_non-redundant_pfms_pfms.txt"
)


def _parse_pfm_text(text: str) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Parse a JASPAR PFM flat file into arrays."""
    pwms, names, matrix_ids = [], [], []

    # Split on header lines
    blocks = re.split(r"(?=^>)", text, flags=re.MULTILINE)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        header = lines[0].lstrip(">").strip()
        parts  = header.split(None, 1)
        mid    = parts[0]
        name   = parts[1] if len(parts) > 1 else ""

        rows = []
        for line in lines[1:]:
            # e.g.  "A  [ 0  3 79 40 ]"  or  "A [0 3 79]"
            nums = re.findall(r"[\d.]+", line)
            if nums:
                rows.append([float(x) for x in nums])

        if len(rows) != 4:
            continue
        mat = np.array(rows, dtype=np.float32)   # [4, L]
        if mat.shape[1] < 3:
            continue

        pwms.append(mat)
        names.append(name)
        matrix_ids.append(mid)

    return pwms, names, matrix_ids


def download_jaspar(out_path: str | Path) -> str:
    """
    Download JASPAR2024 CORE vertebrate PWMs and save to out_path (.json).

    Returns path of the saved file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading JASPAR2024 CORE vertebrates PFMs …\n  {JASPAR_FLAT}")
    resp = requests.get(JASPAR_FLAT, timeout=60)
    resp.raise_for_status()
    print(f"  Downloaded {len(resp.content) / 1024:.0f} KB")

    pwms, names, matrix_ids = _parse_pfm_text(resp.text)
    print(f"  Parsed {len(pwms)} PFMs")

    # Serialise as JSON (lists of lists — no pickle needed)
    data = {
        "pwms":       [m.tolist() for m in pwms],
        "names":      names,
        "matrix_ids": matrix_ids,
    }
    with open(out_path, "w") as f:
        json.dump(data, f)

    print(f"Saved → {out_path}")
    return str(out_path)


def load_jaspar(json_path: str | Path) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Load saved JASPAR JSON. Returns (pwms, names, matrix_ids)."""
    with open(json_path) as f:
        data = json.load(f)
    pwms       = [np.array(m, dtype=np.float32) for m in data["pwms"]]
    names      = data["names"]
    matrix_ids = data["matrix_ids"]
    print(f"Loaded {len(pwms)} JASPAR PWMs from {json_path}")
    return pwms, names, matrix_ids
