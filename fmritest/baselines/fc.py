"""Correlation-FC baselines: masked-region reconstruction and network recovery (PCA / varimax / ICA)."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import FastICA


def group_fc(X: np.ndarray) -> np.ndarray:
    """Pooled correlation matrix over subjects (X: (N, T, R), already z-scored per subject)."""
    Xc = X.reshape(-1, X.shape[-1])
    return np.corrcoef(Xc.T)


def fc_reconstruct(C: np.ndarray, x: np.ndarray, mask: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Conditional-Gaussian (best linear, instantaneous) prediction of masked regions from unmasked ones.

    x: (M, R, W), mask: (M, R). For each sample, x_m = C_mo C_oo^-1 x_o at every timepoint.
    Unmasked entries are returned unchanged.
    """
    pred = x.copy()
    cache = {}
    for i in range(len(x)):
        m = mask[i]
        key = m.tobytes()
        if key not in cache:
            o = ~m
            Coo = C[np.ix_(o, o)] + ridge * np.eye(o.sum())
            cache[key] = C[np.ix_(m, o)] @ np.linalg.inv(Coo)
        pred[i, mask[i]] = cache[key] @ x[i, ~mask[i]]
    return pred


def varimax(L: np.ndarray, gamma: float = 1.0, n_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    p, k = L.shape
    Rm = np.eye(k)
    d = 0.0
    for _ in range(n_iter):
        Lr = L @ Rm
        u, s, vt = np.linalg.svd(L.T @ (Lr**3 - (gamma / p) * Lr @ np.diag((Lr**2).sum(0))))
        Rm = u @ vt
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return L @ Rm


def _orient(C: np.ndarray) -> np.ndarray:
    """Flip each component so its largest-magnitude loading is positive."""
    idx = np.abs(C).argmax(0)
    return C * np.sign(C[idx, np.arange(C.shape[1])])


def components_from_affinity(A: np.ndarray, k: int, method: str = "eig_varimax") -> np.ndarray:
    """Extract k spatial components (R x k) from a symmetric region-by-region affinity matrix.

    No centring: when networks tile the regions, the constant vector lies inside the network
    subspace, so double-centring would discard one network dimension (tests/test_baselines.py).
    Eigenvalues are scaled by sqrt(|w|); clipping negative eigenvalues to 0 yields degenerate
    all-zero components."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    order = np.argsort(w)[::-1][:k]
    L = V[:, order] * np.sqrt(np.abs(w[order]))
    if method == "eig":
        return _orient(L)
    if method == "eig_varimax":
        return _orient(varimax(L))
    raise ValueError(method)


def fc_components(X: np.ndarray, k: int, method: str, seed: int = 0) -> np.ndarray:
    """Network recovery baselines on training data. X: (N, T, R) z-scored."""
    if method in ("eig", "eig_varimax"):
        C = group_fc(X)
        np.fill_diagonal(C, 0.0)  # match the model readout, which has no self-affinity
        return components_from_affinity(C, k, method)
    if method == "pca":
        C = group_fc(X)
        return components_from_affinity(C, k, "eig")
    if method == "ica":
        ica = FastICA(n_components=k, random_state=seed, max_iter=2000, whiten="unit-variance")
        ica.fit(X.reshape(-1, X.shape[-1]))
        return _orient(ica.mixing_)
    raise ValueError(method)


# --------------------------------------------------------------------------------------------
# Stronger linear baselines (added after review): temporal context and site-conditional covariance.
# The best of these, not pooled instantaneous FC, is the bar the model must clear in gate G1.

def lagged_cov(X: np.ndarray, lags: int) -> np.ndarray:
    """Spatio-temporal second-moment matrix of stacked vectors [x(t-L), ..., x(t+L)] (dim R*(2L+1)),
    ordered lag-major. X: (N, T, R), z-scored per subject (so ~zero mean)."""
    N, T, R = X.shape
    D = R * (2 * lags + 1)
    S = np.zeros((D, D))
    n = 0
    for x in X:
        V = np.concatenate([x[l:T - 2 * lags + l] for l in range(2 * lags + 1)], axis=1)
        S += V.T @ V
        n += len(V)
    return S / n


def lagged_reconstruct(S: np.ndarray, lags: int, x: np.ndarray, mask: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    """Best linear prediction of each masked region at time t from all unmasked regions at
    t-L..t+L (truncated at window edges), under the Gaussian with second moment S.
    x: (M, R, W); mask: (M, R). With lags=0 this equals fc_reconstruct."""
    M, R, W = x.shape
    nl = 2 * lags + 1
    pred = x.copy()
    cache = {}
    for i in range(M):
        m = mask[i]
        o_idx = np.flatnonzero(~m)
        m_idx = np.flatnonzero(m)
        for t in range(W):
            lo, hi = max(0, t - lags), min(W - 1, t + lags)
            key = (m.tobytes(), t - lo, hi - t)
            if key not in cache:
                # predictors: unmasked regions at offsets lo-t..hi-t; target: masked regions at offset 0
                offs = range(lo - t, hi - t + 1)
                pi = np.concatenate([(o + lags) * R + o_idx for o in offs])
                ti = lags * R + m_idx
                Spp = S[np.ix_(pi, pi)] + ridge * np.eye(len(pi))
                cache[key] = np.linalg.solve(Spp, S[np.ix_(pi, ti)]).T
            B = cache[key]
            v = x[i][o_idx][:, lo:hi + 1].T.reshape(-1)  # offset-major, matching pi
            pred[i, m_idx, t] = B @ v
    return pred
