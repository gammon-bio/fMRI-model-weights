"""Evaluation metrics: Hungarian-matched network recovery with a shuffled-label null."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _colcorr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation between every column of A (R x k) and every column of B (R x K)."""
    A = (A - A.mean(0)) / np.where(A.std(0) > 0, A.std(0), 1.0)
    B = (B - B.mean(0)) / np.where(B.std(0) > 0, B.std(0), 1.0)
    return A.T @ B / A.shape[0]


def network_recovery(C: np.ndarray, M: np.ndarray) -> dict:
    """Hungarian-match recovered components C (R x k) to ground-truth memberships M (R x K).

    Score = mean over matched pairs of |Pearson r| (absolute: component sign is arbitrary).
    Returns score, per-network r and the assignment (gt network -> component).
    """
    r = np.abs(_colcorr(C, M))  # (k, K)
    rows, cols = linear_sum_assignment(-r)
    per = np.zeros(M.shape[1])
    per[cols] = r[rows, cols]
    assign = -np.ones(M.shape[1], dtype=int)
    assign[cols] = rows
    return {"score": float(r[rows, cols].sum() / M.shape[1]), "per_network": per.tolist(),
            "assignment": assign.tolist()}


def shuffled_label_null(C: np.ndarray, M: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    """Null distribution: permute region labels of the ground truth (rows of M), re-match, re-score."""
    rng = np.random.default_rng(seed)
    return np.array([network_recovery(C, M[rng.permutation(M.shape[0])])["score"] for _ in range(n_perm)])


def recovery_with_null(C, M, n_perm, seed) -> dict:
    res = network_recovery(C, M)
    null = shuffled_label_null(C, M, n_perm, seed)
    res.update(null_mean=float(null.mean()), null_q95=float(np.quantile(null, 0.95)),
               p_perm=float((1 + (null >= res["score"]).sum()) / (1 + n_perm)))
    return res


def mean_sd(xs) -> dict:
    xs = np.asarray(xs, dtype=float)
    return {"mean": float(xs.mean()), "sd": float(xs.std(ddof=1)) if len(xs) > 1 else float("nan"),
            "n": int(len(xs)), "values": xs.tolist()}
