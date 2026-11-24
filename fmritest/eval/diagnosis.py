"""Planted-diagnosis shortcut evaluation.

A diagnosis probe is fit on training subjects, where diagnosis is correlated with site, and
evaluated on three sets:
  test          : held-out subjects from the training sites, correlation intact (in-distribution)
  test_balanced : new subjects from the training sites with 50/50 labels within each site
                  (correlation broken, no site shift) -> isolates shortcut reliance
  test_site     : the held-out site, 50/50 labels (correlation broken AND an unseen scanner)
Reported per set: balanced accuracy and ROC AUC. Shortcut gap = bacc(test) - bacc(test_balanced).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..baselines.leace import LeaceEraser
from ..models.train import to_windows

EVAL_SETS = ("test", "test_balanced", "test_site", "val_balanced")


@torch.no_grad()
def subject_features(insp, Xs: np.ndarray, window: int, kind: str = "token_concat", hook: str | None = None):
    """Subject-level features from unmasked windows, averaged over windows.
    token_concat -> final residual stream per region (n, R*d); pooled -> its mean over regions (n, d);
    attention -> off-diagonal attention patterns averaged over heads, one block per layer (n, L*R*(R-1))."""
    xw = to_windows(Xs, window).astype(np.float32)  # (n, nw, R, W)
    n, nw, R, W = xw.shape
    if kind == "attention":
        names = [f"blocks.{l}.attn.hook_pattern" for l in range(len(insp.model.blocks))]
        _, cache = insp.run_with_cache(torch.from_numpy(xw.reshape(n * nw, R, W)), None, names)
        off = ~np.eye(R, dtype=bool)
        return np.concatenate([cache[nm].numpy().mean(1).reshape(n, nw, R, R).mean(1)[:, off] for nm in names], 1)
    hook = hook or f"blocks.{len(insp.model.blocks) - 1}.hook_resid_post"
    _, cache = insp.run_with_cache(torch.from_numpy(xw.reshape(n * nw, R, W)), None, [hook])
    h = cache[hook].numpy().reshape(n, nw, R, -1).mean(1)  # (n, R, d)
    if kind == "token_concat":
        return h.reshape(n, -1)
    if kind == "pooled":
        return h.mean(1)
    raise ValueError(kind)


def balance_weights(site: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Importance weights making y independent of site in the training set (each site x label cell
    gets equal total weight)."""
    w = np.zeros(len(y), dtype=float)
    for c in np.unique(site):
        for v in np.unique(y):
            m = (site == c) & (y == v)
            if m.any():
                w[m] = 1.0 / m.sum()
    return w * len(y) / w.sum()


def diag_eval(F: dict, y: np.ndarray, site: np.ndarray, sp: dict, C: float, seed: int,
              reweight: bool = False, erase_site: bool = False) -> dict:
    """F: {split_name: features}, aligned with sp[split_name]. Fit on sp['train'] only."""
    Ftr = F["train"]
    ytr, str_ = y[sp["train"]], site[sp["train"]]
    tf = lambda Z: Z
    if erase_site:
        er = LeaceEraser().fit(Ftr, str_)
        tf = er.transform
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=seed))
    kw = {"logisticregression__sample_weight": balance_weights(str_, ytr)} if reweight else {}
    clf.fit(tf(Ftr), ytr, **kw)
    out = {}
    for k in EVAL_SETS:
        if k not in sp or len(sp[k]) == 0:
            continue
        Z, yy = tf(F[k]), y[sp[k]]
        s = clf.decision_function(Z)
        out[k] = {"bacc": float(balanced_accuracy_score(yy, (s > 0).astype(int))),
                  "auc": float(roc_auc_score(yy, s)) if len(np.unique(yy)) == 2 else float("nan"),
                  "n": int(len(yy)), "prevalence": float(yy.mean())}
    if "test" in out and "test_balanced" in out:
        out["shortcut_gap_bacc"] = out["test"]["bacc"] - out["test_balanced"]["bacc"]
    return out


def site_only_reference(y, site, sp, seed) -> dict:
    """A probe that sees only the site label: the pure-shortcut ceiling/floor."""
    oh = np.eye(site.max() + 1)
    F = {k: oh[site[v]] for k, v in sp.items()}
    return diag_eval(F, y, site, sp, C=1.0, seed=seed)
