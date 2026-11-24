"""BrainLM Stage 0c: ground-truth site subspaces in the representations read downstream of layer 9.

Plan: docs/brainlm_stage0c_plan.md (d37c86b). Weight-free helpers (numpy) plus thin torch wrappers
around `HeadModel` (fmritest/brainlm/model.py):

- `energy_basis`: uncentred PCA of per-subject differences D = r_on - r_off; k = smallest number of
  directions reaching >= 90 % of the energy, capped at 8.
- `overlap_basis`: W_alpha = orth(cos(theta) Q_S + sin(theta) Q_R), Q_R Haar-random and orthogonal
  to S, with cos^2(theta) = alpha, so that tr(P_W P_S) / k = alpha.
- `fold_projection`: a reader W acting on an LN output x; W (I - Q Q^T) x == W (x - Q Q^T x).
- `fold_leace`: head(r(x)) with r the LEACE eraser, folded into head weight / bias.
- `metric_m`: m = 1 - ||P_S1 D_edited||^2 / ||P_S1 D_unedited||^2.

Reader inputs of the head model (all LN outputs):
  layer10.attn = layers[0].layernorm_before -> query / key / value
  layer10.mlp  = layers[0].layernorm_after  -> intermediate.dense
  layer11.attn, layer11.mlp likewise for layers[1]; head = pooled final-LN output (= R1).
"""
from __future__ import annotations

import numpy as np

READERS = ("layer10.attn", "layer10.mlp", "layer11.attn", "layer11.mlp", "head")


# ----------------------------------------------------------------------------- numpy core
def energy_basis(D: np.ndarray, thresh: float = 0.90, cap: int = 8) -> tuple[np.ndarray, int, list]:
    """Uncentred PCA of D (n, d). Returns (Q (d, k), k, cumulative energy curve of the first 2*cap PCs)."""
    D = np.asarray(D, dtype=np.float64)
    _, s, Vt = np.linalg.svd(D, full_matrices=False)
    e = s**2 / max(float((s**2).sum()), 1e-300)
    cum = np.cumsum(e)
    k = int(np.searchsorted(cum, thresh - 1e-12) + 1)
    k = min(k, cap)
    return Vt[:k].T.copy(), k, [float(c) for c in cum[: 2 * cap]]


def projector(Q: np.ndarray) -> np.ndarray:
    return Q @ Q.T


def overlap(W: np.ndarray, Q_S: np.ndarray) -> float:
    """tr(P_W P_S) / k for orthonormal W, Q_S of the same width k."""
    return float(np.sum((Q_S.T @ W) ** 2) / Q_S.shape[1])


def overlap_basis(rng: np.random.Generator, Q_S: np.ndarray, alpha: float, tol: float = 1e-6) -> np.ndarray:
    """Orthonormal (d, k) basis with overlap tr(P_W P_S)/k == alpha (checked to `tol`)."""
    d, k = Q_S.shape
    if d < 2 * k:
        raise ValueError("need d >= 2k")
    G = rng.standard_normal((d, k))
    G = G - Q_S @ (Q_S.T @ G)
    Q_R, R = np.linalg.qr(G)
    Q_R = Q_R * np.sign(np.diag(R))
    Q_R = Q_R - Q_S @ (Q_S.T @ Q_R)          # re-orthogonalise against S (numerical hygiene)
    Q_R, _ = np.linalg.qr(Q_R)
    th = np.arccos(np.sqrt(np.clip(alpha, 0.0, 1.0)))
    W, R = np.linalg.qr(np.cos(th) * Q_S + np.sin(th) * Q_R)
    W = W * np.sign(np.diag(R))
    ov = overlap(W, Q_S)
    if abs(ov - alpha) > tol:
        raise AssertionError(f"overlap {ov} != alpha {alpha}")
    return W


def fold_projection(W: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Reader weight (out, in) acting on x: W (I - Q Q^T)."""
    return W - (W @ Q) @ Q.T


def fold_leace(w: np.ndarray, b: np.ndarray, proj: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """head(r(x)) with r(x) = x - proj (x - mu):  w r(x) + b = (w - w proj) x + (b + w proj mu)."""
    wp = w @ proj
    return w - wp, b + wp @ mu


def metric_m(D_edited: np.ndarray, D_unedited: np.ndarray, Q_S1: np.ndarray) -> float:
    """m = 1 - ||P_S1 D_edited||^2 / ||P_S1 D_unedited||^2 (rows = subjects)."""
    num = float(np.sum((np.asarray(D_edited, np.float64) @ Q_S1) ** 2))
    den = float(np.sum((np.asarray(D_unedited, np.float64) @ Q_S1) ** 2))
    return 1.0 - num / den
