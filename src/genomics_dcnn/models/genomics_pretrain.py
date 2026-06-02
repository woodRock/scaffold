"""
Self-supervised masked-nucleotide prediction model for DNA sequences.

Task: given a DNA sequence with 15% of positions masked, predict the original
nucleotide (A/C/G/T) at each masked position.

Trains two components that transfer directly to GenomicsCNN:
  nuc_proj — Linear(vocab_size → cnn_channels): per-position nucleotide encoder
  cnn      — three dilated symmetric ResBlocks: learns local sequence context
             at receptive fields 7 → 19 → 43 bp

Unlike ChromaDCNN's causal next-frame pretraining, masking is bidirectional.
The symmetric ResBlocks used here are identical to those in GenomicsCNN, so
state dicts transfer with no architecture mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class GenomicsPretrainConfig:
    seq_len: int = 251
    vocab_size: int = 4       # ACGT
    cnn_channels: int = 128   # must match GenomicsCNNConfig.cnn_channels
    kernel_size: int = 7
    dropout: float = 0.1


class _ResBlock1d(nn.Module):
    """Symmetric dilated residual block — identical to GenomicsCNN's block."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad        = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.bn1   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.bn2   = nn.BatchNorm1d(channels)
        self.act   = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x   = self.act(self.bn1(self.conv1(x)))
        x   = self.drop(x)
        x   = self.bn2(self.conv2(x))
        return self.act(res + x)


class GenomicsMaskedPredictor(nn.Module):
    """
    Masked-nucleotide predictor for DNA sequences.

    Architecture:
      nuc_proj   — Linear(vocab_size → cnn_channels)
      pos_embed  — learned positional embedding
      cnn        — three symmetric dilated ResBlocks (dilation 1 / 2 / 4)
      pred_head  — Linear(cnn_channels → vocab_size)

    Pretraining objective: cross-entropy at masked positions only.

    Weight transfer to GenomicsCNN:
      model.nuc_proj → GenomicsCNN.nuc_proj   (direct)
      model.cnn[i].* → GenomicsCNN.cnn[i].*   (identical Conv1d shapes)
    """

    def __init__(self, config: GenomicsPretrainConfig) -> None:
        super().__init__()
        ch = config.cnn_channels
        self.nuc_proj  = nn.Linear(config.vocab_size, ch)
        self.pos_embed = nn.Embedding(config.seq_len, ch)

        k, d = config.kernel_size, config.dropout
        self.cnn = nn.Sequential(
            _ResBlock1d(ch, k, dilation=1, dropout=d),
            _ResBlock1d(ch, k, dilation=2, dropout=d),
            _ResBlock1d(ch, k, dilation=4, dropout=d),
        )
        self.pred_head = nn.Linear(ch, config.vocab_size)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x    : [B, L, vocab_size]  one-hot DNA (masked positions already zeroed)
        mask : [B, L]              bool, True = masked position

        Returns
        -------
        logits : [M, vocab_size]  predictions at masked positions only
        """
        B, L, V = x.shape
        h = self.nuc_proj(x.reshape(B * L, V)).view(B, L, -1)         # [B, L, ch]
        h = h + self.pos_embed(torch.arange(L, device=x.device))      # [B, L, ch]
        h = self.cnn(h.transpose(1, 2)).transpose(1, 2)               # [B, L, ch]
        return self.pred_head(h[mask])                                  # [M, vocab_size]
