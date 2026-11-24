"""End-to-end diagnosis fine-tuning: puts the site shortcut *inside* the model (Phase 2 repair target).

A pretrained masked-reconstruction backbone gets a linear head on the region-mean of its final
(layer-normed) residual stream and is fine-tuned on the planted diagnosis with binary cross-entropy,
using training subjects only, where diagnosis is correlated with site. A shortcut model should then
score well in-distribution and fail when the correlation is broken. The same fine-tune with
site x label importance weights (`reweight=True`) is the reference for what a non-shortcut model of
the same architecture and data can reach (repair headroom).

Evaluation is at the subject level: logits are averaged over a subject's non-overlapping windows.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .eval.diagnosis import EVAL_SETS, balance_weights
from .models.train import set_seed, to_windows
from .models.transformer import MaskedRegionTransformer


class DiagnosisModel(MaskedRegionTransformer):
    """The pretrained transformer plus a linear diagnosis head. forward(x, mask=None) -> logits (B,).
    Subclassing keeps every hook name identical to the backbone's (blocks.{l}..., hook_embed), so the
    interp API and readouts work unchanged. The head reads ln_final(residual) averaged over regions;
    the reconstruction unembed is kept but unused."""

    @classmethod
    def from_backbone(cls, backbone: MaskedRegionTransformer) -> "DiagnosisModel":
        b = backbone
        m = cls(b.n_regions, b.window, b.region_emb.shape[1], b.blocks[0].attn.h, len(b.blocks),
                b.blocks[0].mlp.W_in.out_features)
        m.load_state_dict(b.state_dict(), strict=False)
        return m

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.head = nn.Linear(self.region_emb.shape[1], 1)

    def embed_subject(self, x, mask=None):
        h = self.embed(x)
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_emb.expand_as(h), h)
        h = self.hook_embed(h + self.region_emb)
        for blk in self.blocks:
            h = blk(h)
        return self.ln_final(h).mean(1)

    def forward(self, x, mask=None):
        return self.head(self.embed_subject(x, mask)).squeeze(-1)

    @property
    def backbone(self):  # backwards-compatible accessor
        return self


def finetune(backbone: MaskedRegionTransformer, X: np.ndarray, y: np.ndarray, site: np.ndarray,
             train_idx: np.ndarray, fcfg: dict, seed: int, n_threads: int, reweight: bool = False):
    """Fine-tune a copy of `backbone` on random windows of training subjects. Returns (model, curve)."""
    set_seed(seed + 555, n_threads)
    torch.manual_seed(seed + 557)  # head init
    model = DiagnosisModel.from_backbone(backbone).train()
    W = backbone.window
    w_subj = balance_weights(site[train_idx], y[train_idx]) if reweight else np.ones(len(train_idx))
    head_ids = {id(p) for p in model.head.parameters()}
    groups = [{"params": list(model.head.parameters()), "lr": fcfg["lr_head"]},
              {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": fcfg["lr_backbone"]}]
    opt = torch.optim.AdamW(groups, weight_decay=fcfg["weight_decay"])
    rng = np.random.default_rng(seed + 556)
    T = X.shape[1]
    curve, run, t0 = [], 0.0, time.time()
    for step in range(fcfg["steps"]):
        s = rng.integers(0, len(train_idx), fcfg["batch_size"])
        t = rng.integers(0, T - W + 1, fcfg["batch_size"])
        xb = torch.from_numpy(np.stack([X[train_idx[i], tt:tt + W].T for i, tt in zip(s, t)]).astype(np.float32))
        yb = torch.from_numpy(y[train_idx[s]].astype(np.float32))
        wb = torch.from_numpy(w_subj[s].astype(np.float32))
        loss = (F.binary_cross_entropy_with_logits(model(xb), yb, reduction="none") * wb).sum() / wb.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run += float(loss.detach())
        if (step + 1) % 100 == 0:
            curve.append({"step": step + 1, "loss": run / 100})
            run = 0.0
    model.eval()
    return model, {"curve": curve, "seconds": time.time() - t0}


@torch.no_grad()
def subject_logits(model: DiagnosisModel, Xs: np.ndarray) -> np.ndarray:
    xw = to_windows(Xs, model.window).astype(np.float32)
    n, nw, R, W = xw.shape
    return model(torch.from_numpy(xw.reshape(n * nw, R, W))).numpy().reshape(n, nw).mean(1)


@torch.no_grad()
def subject_embeddings(model: DiagnosisModel, Xs: np.ndarray) -> np.ndarray:
    xw = to_windows(Xs, model.window).astype(np.float32)
    n, nw, R, W = xw.shape
    return model.embed_subject(torch.from_numpy(xw.reshape(n * nw, R, W))).numpy().reshape(n, nw, -1).mean(1)


def evaluate(model: DiagnosisModel, X: np.ndarray, y: np.ndarray, sp: dict) -> dict:
    out = {}
    for k in EVAL_SETS:
        if k not in sp:
            continue
        s = subject_logits(model, X[sp[k]])
        yy = y[sp[k]]
        out[k] = {"bacc": float(balanced_accuracy_score(yy, (s > 0).astype(int))),
                  "auc": float(roc_auc_score(yy, s)), "n": int(len(yy))}
    if "test" in out and "test_balanced" in out:
        out["shortcut_gap_bacc"] = out["test"]["bacc"] - out["test_balanced"]["bacc"]
    return out
