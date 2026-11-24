"""Comparators, all expressed as edits of the vanilla diagnosis model and scored by the same Scorer.

Post-hoc fixes as head replacements. The head computes logit = w . e + b on
e = mean_r LN_final(resid)[r], and the subject logit is the mean over windows, which is affine in e,
so any affine feature map followed by a linear probe on subject-mean embeddings is absorbable into
the head exactly:
    LEACE      r(e) = e - P (e - mu)
    scaler     z = (r - m) / s
    probe      logit = c . z + c0
    =>  w' = (I - P)^T (c / s),   b' = c0 - (c / s) . m + (c / s) . (P mu)
(`leace_probe_head`; equivalence tested against the sklearn pipeline in tests/test_harness.py).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..baselines.leace import LeaceEraser
from .modelio import HEAD_IN, LOGIT, edited_in_place, oracle_readout


def leace_probe_head(E: np.ndarray, site: np.ndarray, y: np.ndarray, C: float, erase: bool = True):
    """Fit (LEACE-site ->) StandardScaler -> L2 logistic on subject embeddings E; return the
    equivalent head (w (d,), b float) and the fitted parts. lbfgs is deterministic, so the fit is a
    pure function of (E, site, y, C)."""
    E = np.asarray(E, dtype=np.float64)
    er = LeaceEraser().fit(E, site) if erase else None
    Z = er.transform(E) if erase else E
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=0))
    clf.fit(Z, y)
    sc, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
    c = lr.coef_[0] / sc.scale_
    b = float(lr.intercept_[0] - c @ sc.mean_)
    w = c.copy()
    if erase:
        w = c - er.proj.T @ c
        b = b + float(c @ (er.proj @ er.mu))
    return w, b, {"eraser": er, "clf": clf}


def head_edit(w: np.ndarray, b: float) -> dict:
    return {"head.weight": np.asarray(w, dtype=np.float32).reshape(1, -1),
            "head.bias": np.asarray([b], dtype=np.float32)}


class BudgetAccount:
    """Oracle accounting in subject x hook units; the logit always counts as one hook."""

    def __init__(self, budget: int, used: int = 0):
        self.budget, self.used = int(budget), int(used)

    @staticmethod
    def cost(n_subjects: int, n_hooks: int) -> int:
        return int(n_subjects) * (int(n_hooks) + 1)

    def can(self, c: int) -> bool:
        return self.used + c <= self.budget

    def charge(self, c: int) -> None:
        if not self.can(c):
            raise RuntimeError(f"budget exceeded: used {self.used} + {c} > {self.budget}")
        self.used += c


def cell_balanced_acc(s: np.ndarray, y: np.ndarray, site: np.ndarray) -> float:
    """Mean accuracy over the (site, label) cells present: the shortcut-insensitive practice score."""
    accs = [float(((s[m] > 0).astype(int) == y[m]).mean())
            for c in np.unique(site) for v in (0, 1) if (m := (site == c) & (y == v)).any()]
    return float(np.mean(accs))


def stratified_subset(y: np.ndarray, site: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Up to m positions, round-robin over (site, label) cells so every cell is represented."""
    cells = [list(rng.permutation(np.flatnonzero((site == c) & (y == v)))) for c in np.unique(site) for v in (0, 1)]
    cells = [c for c in cells if c]
    out = []
    while len(out) < m and any(cells):
        for c in cells:
            if c and len(out) < m:
                out.append(c.pop())
    return np.sort(np.array(out, dtype=int))


def best_of_n(model, X_pool: np.ndarray, y_pool: np.ndarray, site_pool: np.ndarray, budget: int,
              bcfg: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    """Non-LLM brute force at the agent's oracle budget: random rank-1 edits of the head and the last
    MLP W_out, each scored through the oracle (logit only, m subjects -> m units) on a fixed
    site x label stratified subset of the agent pool by cell-balanced accuracy; keep the best edit
    if it beats the unedited model. Returns (edit or {}, info)."""
    acct = BudgetAccount(budget)
    L = len(model.blocks)
    tensors = [t.replace("{last}", str(L - 1)) for t in bcfg["tensors"]]
    m = int(bcfg["subjects_per_eval"])
    if budget < 2 * m:
        m = budget // 2
    n_cells = len(np.unique(site_pool)) * 2
    info = {"budget": int(budget), "subjects_per_eval": int(m), "tensors": tensors, "criterion": "cell_balanced_acc"}
    if m < n_cells:
        info.update(n_candidates=0, used=0, note="budget too small for one baseline + one candidate")
        return {}, info
    sel = stratified_subset(y_pool, site_pool, m, rng)
    Xs, ys, ss = X_pool[sel], y_pool[sel], site_pool[sel]
    sd = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}

    def evaluate(mdl):
        acct.charge(BudgetAccount.cost(len(sel), 0))
        return cell_balanced_acc(oracle_readout(mdl, Xs, [])[LOGIT], ys, ss)

    best_score, best_edit, n_cand, hist = evaluate(model), {}, 0, []
    base = best_score
    lo, hi = bcfg["rel_norm_range"]
    while acct.can(BudgetAccount.cost(len(sel), 0)):
        t = tensors[rng.integers(len(tensors))]
        W = sd[t]
        u = rng.standard_normal(W.shape[0])
        v = rng.standard_normal(W.shape[1])
        D = np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v))
        D *= np.exp(rng.uniform(np.log(lo), np.log(hi))) * np.linalg.norm(W)
        cand = {t: (W + D).astype(np.float32)}
        with edited_in_place(model, cand):
            sc = evaluate(model)
        n_cand += 1
        hist.append(sc)
        if sc > best_score:
            best_score, best_edit = sc, cand
    info.update(n_candidates=n_cand, used=acct.used, baseline_practice=base, best_practice=best_score,
                kept_edit=bool(best_edit), candidate_practice=hist)
    return best_edit, info


def embeddings(model, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return oracle_readout(model, X[idx], [HEAD_IN])[HEAD_IN]
