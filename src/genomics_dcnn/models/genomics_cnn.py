"""
GenomicsCNN: 1D dilated CNN classifier for DNA sequences.

Architecture:
  1. Per-position nucleotide projection  Linear(vocab_size → cnn_channels)
  2. Three dilated 1D ResBlocks (kernel=7, dilation=1/2/4)
       receptive field: 7 → 19 → 43 bp
  3. Dual pooling: global max-pool ∥ soft attention-pool → [B, 2·cnn_channels]
  4. Linear classification head

Directly analogous to ChromatogramCNN from chroma-dcnn:
  mz_max      → vocab_size  (1000 m/z channels → 4 nucleotides)
  n_bins      → seq_len     (200 RT bins → 251 bp)
  num_classes = 2 for human_nontata_promoters (binary)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GenomicsCNNConfig:
    vocab_size: int = 4
    cnn_channels: int = 128
    kernel_size: int = 7
    num_classes: int = 2
    dropout: float = 0.3


class _ResBlock1d(nn.Module):
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


class _AttentionPool1d(nn.Module):
    """Soft attention pooling over the sequence dimension.  x: [B,C,L] → [B,C]"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x_t = x.transpose(1, 2)                    # [B, L, C]
        w   = F.softmax(self.score(x_t), dim=1)    # [B, L, 1]
        return (x_t * w).sum(dim=1)                # [B, C]


class GenomicsCNN(nn.Module):
    """
    Parameters
    ----------
    config    : GenomicsCNNConfig
    condition : ignored — kept for API compatibility with ChromatogramCNN
    """

    def __init__(self, config: GenomicsCNNConfig, condition: str = "from_scratch") -> None:
        super().__init__()
        ch = config.cnn_channels
        self.nuc_proj  = nn.Linear(config.vocab_size, ch)

        k, d = config.kernel_size, config.dropout
        self.cnn = nn.Sequential(
            _ResBlock1d(ch, k, dilation=1, dropout=d),
            _ResBlock1d(ch, k, dilation=2, dropout=d),
            _ResBlock1d(ch, k, dilation=4, dropout=d),
        )
        self.attn_pool = _AttentionPool1d(ch)
        self.head      = nn.Linear(ch * 2, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """[B, L, vocab_size] → [B, num_classes]"""
        B, L, V = x.shape
        emb    = self.nuc_proj(x.reshape(B * L, V)).view(B, L, -1)   # [B, L, ch]
        x      = emb.transpose(1, 2)                                   # [B, ch, L]
        x      = self.cnn(x)
        x_max  = x.max(dim=-1).values                                  # [B, ch]
        x_attn = self.attn_pool(x)                                     # [B, ch]
        return self.head(torch.cat([x_max, x_attn], dim=-1))

    def load_pretrained_genomics_encoder(self, checkpoint_path: str) -> None:
        """Load nuc_proj and CNN weights from a GenomicsMaskedPredictor checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.nuc_proj.load_state_dict(ckpt["nuc_proj_state"])
        print(f"  Loaded pretrained nuc_proj from {checkpoint_path}")
        if "cnn_state" in ckpt:
            self.cnn.load_state_dict(ckpt["cnn_state"])
            print(f"  Loaded pretrained CNN from {checkpoint_path}")
