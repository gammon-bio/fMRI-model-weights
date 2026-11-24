"""Subject- and site-level splits with explicit leakage checks.

Splits (all at the subject level -- a subject's whole scan lives in exactly one split):
  train      : subjects from the training sites, used to fit the model, FC baseline, probes, erasers
  val        : subjects from the training sites, used only for monitoring the training curve
  test       : held-out subjects from the training sites (site decoding is evaluated here)
  test_site  : every subject from the held-out site (never seen by anything that is fitted)
  test_balanced : (only with the planted diagnosis) extra subjects from the training sites whose
                  diagnosis is balanced within site, i.e. the label-site correlation is broken
  val_balanced  : same construction, separate subjects, for calibration / model selection only
Within each training site the main cohort is stratified by `strata` (e.g. diagnosis) if given.
"""
from __future__ import annotations

import numpy as np


def make_splits(site: np.ndarray, heldout_site: int, frac=(0.6, 0.2, 0.2), seed: int = 0,
                group: np.ndarray | None = None, strata: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed + 7919)
    group = np.zeros(len(site), dtype=int) if group is None else group
    strata = np.zeros(len(site), dtype=int) if strata is None else strata
    out = {"train": [], "val": [], "test": [], "test_site": []}
    if np.any(group == 1):
        out["test_balanced"] = np.flatnonzero(group == 1).tolist()
    if np.any(group == 2):
        out["val_balanced"] = np.flatnonzero(group == 2).tolist()
    for c in np.unique(site):
        if c == heldout_site:
            out["test_site"].extend(np.flatnonzero((site == c) & (group == 0)).tolist())
            continue
        for st in np.unique(strata):
            idx = rng.permutation(np.flatnonzero((site == c) & (group == 0) & (strata == st)))
            n_tr = int(round(frac[0] * len(idx)))
            n_va = int(round(frac[1] * len(idx)))
            out["train"].extend(idx[:n_tr].tolist())
            out["val"].extend(idx[n_tr:n_tr + n_va].tolist())
            out["test"].extend(idx[n_tr + n_va:].tolist())
    splits = {k: np.array(sorted(v), dtype=int) for k, v in out.items()}
    check_no_leakage(splits, site, heldout_site)
    return splits


def check_no_leakage(splits: dict, site: np.ndarray, heldout_site: int) -> None:
    """Raise if any subject appears in two splits or the held-out site leaks into a fitted split."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = np.intersect1d(splits[a], splits[b])
            if inter.size:
                raise AssertionError(f"subject leakage between {a} and {b}: {inter[:5]}")
    for k in ("train", "val", "test", "test_balanced", "val_balanced"):
        if k in splits and np.any(site[splits[k]] == heldout_site):
            raise AssertionError(f"held-out site {heldout_site} present in split {k}")
    if not np.all(site[splits["test_site"]] == heldout_site):
        raise AssertionError("test_site split contains subjects from a training site")
