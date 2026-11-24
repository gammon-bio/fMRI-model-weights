"""LEACE: closed-form least-squares linear concept erasure (Belrose et al., 2023, arXiv:2306.03819).

Given features x and a categorical concept z (one-hot), the eraser
    r(x) = x - W^+ P W (x - mu),    W = Sigma_xx^{-1/2},  P = orthogonal projector onto colspace(W Sigma_xz)
makes Cov(r(x), z) = 0 (so no linear classifier beats the constant predictor in-sample) while
moving x as little as possible in the whitened norm.
"""
from __future__ import annotations

import numpy as np


class LeaceEraser:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def fit(self, X: np.ndarray, z: np.ndarray) -> "LeaceEraser":
        X = np.asarray(X, dtype=np.float64)
        classes = np.unique(z)
        Z = (np.asarray(z)[:, None] == classes[None, :]).astype(np.float64)
        self.mu = X.mean(0)
        Xc = X - self.mu
        Zc = Z - Z.mean(0)
        n = len(X)
        Sxx = Xc.T @ Xc / n
        Sxz = Xc.T @ Zc / n
        w, V = np.linalg.eigh(Sxx)
        keep = w > self.eps * w.max()
        W = (V[:, keep] / np.sqrt(w[keep])) @ V[:, keep].T
        W_pinv = (V[:, keep] * np.sqrt(w[keep])) @ V[:, keep].T
        U, s, _ = np.linalg.svd(W @ Sxz, full_matrices=False)
        U = U[:, s > self.eps * max(s.max(), 1e-300)]
        P = U @ U.T
        self.proj = W_pinv @ P @ W         # x -> x - proj (x - mu)
        self.rank = U.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X - (X - self.mu) @ self.proj.T
