"""
Self-supervised pretraining of GenomicsMaskedPredictor on synthetic DNA sequences.

Trains with masked nucleotide prediction (MNP):
  - 15% of positions are randomly zeroed in the input
  - model predicts the original nucleotide (A/C/G/T) at each masked position
  - loss: cross-entropy on masked positions only

Uses iteration-based training: each step draws a fresh random batch of synthetic
sequences generated from JASPAR PWMs, giving effectively unlimited variety without
epoch bookkeeping.  No labelled benchmark data is used — no leakage into CV folds.

Saves a checkpoint containing:
  nuc_proj_state — weights for GenomicsCNN.nuc_proj
  cnn_state      — weights for GenomicsCNN.cnn
  config         — GenomicsPretrainConfig
  n_iterations   — total gradient steps taken
  final_loss     — best loss seen
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from genomics_dcnn.models.genomics_pretrain import (
    GenomicsMaskedPredictor,
    GenomicsPretrainConfig,
)

VOCAB_SIZE = 4


def _sample_motif(pwm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Sample one DNA sequence from a PWM using vectorised inverse-CDF sampling.
    Replaces a Python loop over motif positions with a single numpy operation.

    pwm : [4, L]  position frequency matrix
    returns [L] int64 nucleotide indices
    """
    probs    = pwm / (pwm.sum(axis=0, keepdims=True) + 1e-8)   # [4, L]
    cumprobs = np.cumsum(probs, axis=0)                          # [4, L]
    u        = rng.random(pwm.shape[1])                          # [L]
    return (u[None, :] > cumprobs).sum(axis=0).clip(0, 3).astype(np.int64)


def _make_batch(
    pwms: list[np.ndarray],
    batch_size: int,
    seq_len: int,
    n_motifs_range: tuple[int, int],
    mask_ratio: float,
    rng: np.random.Generator,
    bg_probs: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a batch of synthetic DNA sequences with motif-priority masking.

    Background is sampled from a biased nucleotide distribution (default: ~41% GC,
    matching human genome composition) rather than uniform random.  K JASPAR motifs
    are injected per sequence.  Masking fills its budget from injected-motif positions
    first, then random background — concentrating gradient signal on learnable regions.

    Returns
    -------
    x_masked : [B, L, 4]  float32 — input with masked positions zeroed
    targets  : [B, L]     int64   — original nucleotide indices (0 at unmasked)
    mask     : [B, L]     bool    — True at masked positions
    """
    if bg_probs is None:
        # Approximate human genome base composition: ~41% GC (A=T=0.295, C=G=0.205)
        bg_probs = np.array([0.295, 0.205, 0.205, 0.295], dtype=np.float64)

    lo, hi = n_motifs_range

    # Background: biased multinomial instead of i.i.d. uniform
    seqs = np.stack([
        rng.choice(VOCAB_SIZE, size=seq_len, p=bg_probs)
        for _ in range(batch_size)
    ]).astype(np.int64)

    motif_pos_sets: list[set[int]] = [set() for _ in range(batch_size)]

    for b in range(batch_size):
        K = int(rng.integers(lo, hi + 1))
        for _ in range(K):
            pwm = pwms[int(rng.integers(len(pwms)))]
            L_m = pwm.shape[1]
            if L_m >= seq_len:
                continue
            pos = int(rng.integers(0, seq_len - L_m))
            seqs[b, pos:pos + L_m] = _sample_motif(pwm, rng)
            motif_pos_sets[b].update(range(pos, pos + L_m))

    # One-hot encode: [B, L, 4]
    one_hot = np.eye(VOCAB_SIZE, dtype=np.float32)[seqs]

    # Motif-priority masking: fill budget from motif positions first, then background
    n_mask  = max(1, int(seq_len * mask_ratio))
    mask    = np.zeros((batch_size, seq_len), dtype=bool)
    targets = seqs.copy()

    for b in range(batch_size):
        motif_pos = np.fromiter(motif_pos_sets[b], dtype=np.int64)
        bg_pos    = np.array([i for i in range(seq_len) if i not in motif_pos_sets[b]], dtype=np.int64)

        chosen: list[np.ndarray] = []
        if motif_pos.size > 0:
            n_motif = min(motif_pos.size, n_mask)
            chosen.append(rng.choice(motif_pos, size=n_motif, replace=False))
        n_remaining = n_mask - sum(a.size for a in chosen)
        if n_remaining > 0 and bg_pos.size > 0:
            chosen.append(rng.choice(bg_pos, size=min(n_remaining, bg_pos.size), replace=False))

        mask[b, np.concatenate(chosen) if chosen else np.array([], dtype=np.int64)] = True

    x_masked       = one_hot.copy()
    x_masked[mask] = 0.0

    mask_t    = torch.from_numpy(mask)
    targets_t = torch.from_numpy(targets)
    targets_t[~mask_t] = 0   # only care about masked positions in loss

    return (
        torch.from_numpy(x_masked),
        targets_t,
        mask_t,
    )


def pretrain_genomics(
    cfg: dict,
    pwms: list[np.ndarray],
    device: torch.device,
) -> str:
    """
    Train GenomicsMaskedPredictor and save checkpoint.

    Parameters
    ----------
    cfg    : pretraining YAML config dict
    pwms   : list of [4, L_i] float32 JASPAR position frequency matrices
    device : compute device

    Returns
    -------
    Path to saved checkpoint.
    """
    pcfg = cfg["pretraining"]
    mcfg = cfg["model"]

    config = GenomicsPretrainConfig(
        vocab_size   = mcfg.get("vocab_size", 4),
        cnn_channels = mcfg.get("cnn_channels", 128),
        kernel_size  = mcfg.get("kernel_size", 7),
        dropout      = mcfg.get("dropout", 0.1),
    )

    model = GenomicsMaskedPredictor(config).to(device)

    n_iterations    = pcfg["n_iterations"]
    batch_size      = pcfg.get("batch_size", 32)
    lr              = pcfg.get("lr", 1e-3)
    wd              = pcfg.get("weight_decay", 1e-4)
    log_every       = pcfg.get("log_every", 500)
    mask_ratio      = pcfg.get("mask_ratio", 0.15)
    n_motifs_range  = tuple(cfg["data"].get("n_motifs_range", [1, 6]))

    raw_bg = cfg["data"].get("bg_probs")
    bg_probs = np.array(raw_bg, dtype=np.float64) if raw_bg is not None else None

    opt   = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = CosineAnnealingLR(opt, T_max=n_iterations, eta_min=lr * 0.01)
    rng   = np.random.default_rng()

    effective_bg = bg_probs if bg_probs is not None else np.array([0.295, 0.205, 0.205, 0.295])
    print(f"\nPretraining GenomicsMaskedPredictor (masked nucleotide prediction)", flush=True)
    print(f"  JASPAR PWMs   : {len(pwms)}", flush=True)
    print(f"  CNN channels  : {config.cnn_channels}", flush=True)
    print(f"  Mask ratio    : {mask_ratio}  (motif positions prioritised)", flush=True)
    print(f"  n_motifs      : {n_motifs_range}", flush=True)
    print(f"  bg_probs ACGT : {effective_bg.tolist()}", flush=True)
    print(f"  Iterations    : {n_iterations}", flush=True)
    print(f"  Batch size    : {batch_size}", flush=True)
    print(f"  Device        : {device}\n", flush=True)

    best_loss    = float("inf")
    running_loss = 0.0

    model.train()
    for it in range(1, n_iterations + 1):
        x_masked, targets, mask = _make_batch(
            pwms, batch_size, mcfg["seq_len"], n_motifs_range, mask_ratio, rng, bg_probs,
        )
        x_masked = x_masked.to(device)
        targets  = targets.to(device)
        mask     = mask.to(device)

        logits = model(x_masked, mask)                      # [M, vocab_size]
        loss   = nn.functional.cross_entropy(logits, targets[mask])

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), pcfg.get("grad_clip", 1.0))
        opt.step()
        sched.step()

        running_loss += loss.item()

        if it % log_every == 0 or it == n_iterations:
            avg = running_loss / log_every
            if avg < best_loss:
                best_loss = avg
            print(f"  iter {it:>6d}/{n_iterations}  loss={avg:.4f}  best={best_loss:.4f}", flush=True)
            running_loss = 0.0

    ckpt_dir  = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    torch.save({
        "nuc_proj_state": model.nuc_proj.state_dict(),
        "cnn_state":      model.cnn.state_dict(),
        "config":         config,
        "n_iterations":   n_iterations,
        "final_loss":     best_loss,
    }, ckpt_path)

    print(f"\nCheckpoint saved → {ckpt_path}  (best loss={best_loss:.4f})", flush=True)
    return str(ckpt_path)
