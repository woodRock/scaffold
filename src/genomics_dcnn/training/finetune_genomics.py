"""
Fine-tuning and cross-validated evaluation for GenomicsCNN.

Two conditions evaluated identically to chroma-dcnn:

  from_scratch      — random weights, trained end-to-end.

  genomics_pretrain — same architecture, nuc_proj + CNN initialised from
                      masked-nucleotide prediction pretraining on JASPAR sequences.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from genomics_dcnn.data.datasets import DNADataset
from genomics_dcnn.models.genomics_cnn import GenomicsCNN, GenomicsCNNConfig

ConditionName = Literal["from_scratch", "genomics_pretrain"]


def _build_model(
    config: GenomicsCNNConfig,
    condition: ConditionName,
    pretrain_ckpt: str | None = None,
) -> GenomicsCNN:
    model = GenomicsCNN(config)
    if condition == "genomics_pretrain" and pretrain_ckpt:
        model.load_pretrained_genomics_encoder(pretrain_ckpt)
    return model


@torch.no_grad()
def _eval_loss(model, loader, criterion, device):
    total = 0.0
    for x, y in loader:
        total += criterion(model(x.to(device)), y.to(device)).item()
    return total / max(len(loader), 1)


@torch.no_grad()
def _compute_metrics(model, loader, device) -> dict[str, float]:
    all_preds, all_labels = [], []
    for x, y in loader:
        preds = model(x.to(device)).argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return {
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "macro_f1":          f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def _class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    classes, counts = np.unique(labels, return_counts=True)
    w = np.zeros(num_classes, dtype=np.float32)
    for c, n in zip(classes, counts):
        w[c] = len(labels) / (len(classes) * n)
    return torch.tensor(w)


def _train_fold(
    model: GenomicsCNN,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    device: torch.device,
    lr: float | None = None,
) -> dict[str, float]:
    tcfg       = cfg["finetuning"]
    epochs     = tcfg["epochs"]
    batch_size = min(tcfg.get("batch_size", 32), len(X_train))

    train_ds     = DNADataset(X_train, y_train)
    val_ds       = DNADataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    if lr is None:
        lr = tcfg.get("lr", 1e-3)
    wd     = tcfg.get("weight_decay", 0.01)
    params = [p for p in model.parameters() if p.requires_grad]
    opt    = AdamW(params, lr=lr, weight_decay=wd)
    sched  = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(y_train, cfg["task"]["num_classes"]).to(device)
    )

    best_val_loss = float("inf")
    best_state    = copy.deepcopy(model.state_dict())
    stale, patience = 0, tcfg.get("early_stopping_patience", 10)

    for _ in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            criterion(model(x), y).backward()
            nn.utils.clip_grad_norm_(params, tcfg.get("grad_clip", 1.0))
            opt.step()
        sched.step()

        model.eval()
        val_loss = _eval_loss(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return _compute_metrics(model, val_loader, device)


class GenomicsFinetuner:
    def __init__(
        self,
        config: dict,
        X: np.ndarray,
        y: np.ndarray,
        device: str | None = None,
        label_fraction: float = 1.0,
    ) -> None:
        self.cfg            = config
        self.X              = X
        self.y              = y
        self.label_fraction = label_fraction

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        m = config["model"]
        self.model_config = GenomicsCNNConfig(
            vocab_size   = m.get("vocab_size", 4),
            cnn_channels = m.get("cnn_channels", 128),
            kernel_size  = m.get("kernel_size", 7),
            num_classes  = config["task"]["num_classes"],
            dropout      = m.get("dropout", 0.3),
        )
        self.pretrain_ckpt = (
            config.get("pretrained_checkpoints", {}).get("genomics_pretrain")
        )

    def evaluate_condition(
        self,
        condition: ConditionName,
        seeds: list[int] | None = None,
    ) -> dict[str, list[float]]:
        if seeds is None:
            seeds = self.cfg["task"].get("cv_seeds", list(range(10)))

        tcfg = self.cfg["finetuning"]
        lr   = (
            tcfg.get("lr_scratch", tcfg.get("lr", 1e-3))
            if condition == "from_scratch"
            else tcfg.get("lr", 1e-3)
        )

        all_ba, all_f1 = [], []
        n_folds = self.cfg["task"].get("cv_folds", 5)

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for train_idx, val_idx in skf.split(self.X, self.y):
                # Subsample training split only — val split is always full
                if self.label_fraction < 1.0:
                    rng = np.random.default_rng(seed)
                    n_keep = max(1, int(len(train_idx) * self.label_fraction))
                    train_idx = rng.choice(train_idx, size=n_keep, replace=False)

                model = _build_model(
                    self.model_config, condition, self.pretrain_ckpt,
                ).to(self.device)
                metrics = _train_fold(
                    model,
                    self.X[train_idx], self.X[val_idx],
                    self.y[train_idx], self.y[val_idx],
                    self.cfg, self.device, lr=lr,
                )
                all_ba.append(metrics["balanced_accuracy"])
                all_f1.append(metrics["macro_f1"])

        return {"balanced_accuracy": all_ba, "macro_f1": all_f1}

    def run_all_conditions(self, seeds: list[int] | None = None) -> dict[str, dict]:
        conditions: list[ConditionName] = ["from_scratch"]
        if self.pretrain_ckpt and Path(self.pretrain_ckpt).exists():
            conditions.append("genomics_pretrain")

        results = {}
        for cond in conditions:
            print(f"\n--- Condition: {cond} ---")
            results[cond] = self.evaluate_condition(cond, seeds)
            ba = results[cond]["balanced_accuracy"]
            print(f"  balanced_accuracy: {np.mean(ba):.3f} ± {np.std(ba):.3f}"
                  f"  (n={len(ba)} fold×seed runs)")
        return results
