"""Carrier subspace on the TEMPORAL axis of BrainLM's patch-embedding kernel.

The kernel `vit.embeddings.patch_embeddings.projection.weight` has shape [768, 3, 16, 16]. From the
tutorial packing (see pack.py) time runs along the image width, i.e. the kernel's last axis. Each
output channel's response to a patch is <W[o], patch>; removing the carrier subspace span(Q) from
every 16-sample temporal row of W (all output channels, input channels and parcel rows) makes the
embedding blind to any input whose temporal profile lies in span(Q):
    W <- W - (W Q) Q^T   along the time axis.
"""
from __future__ import annotations

import numpy as np

from .pack import KERNEL_TIME_AXIS, PATCH

TR = 0.72


def carrier_basis(freqs, n: int = PATCH, tr: float = TR) -> np.ndarray:
    """Orthonormal basis (n, 2 * len(freqs)) of sin/cos at `freqs` Hz sampled at n points, spacing tr."""
    t = np.arange(n) * tr
    cols = []
    for f in np.atleast_1d(freqs):
        cols += [np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)]
    Q, R = np.linalg.qr(np.stack(cols, axis=1))
    if np.min(np.abs(np.diag(R))) < 1e-8:
        raise ValueError("carrier vectors are linearly dependent at this resolution")
    return Q


def random_basis(rng: np.random.Generator, k: int, n: int = PATCH) -> np.ndarray:
    """Haar-random orthonormal (n, k) basis."""
    Q, R = np.linalg.qr(rng.standard_normal((n, k)))
    return Q * np.sign(np.diag(R))


def project_out(W, Q, axis: int = KERNEL_TIME_AXIS):
    """Return W with span(Q) removed along `axis` (works for numpy arrays and torch tensors)."""
    try:
        import torch
        if isinstance(W, torch.Tensor):
            Qt = torch.as_tensor(Q, dtype=W.dtype, device=W.device)
            Wm = torch.movedim(W, axis, -1)
            Wm = Wm - (Wm @ Qt) @ Qt.T
            return torch.movedim(Wm, -1, axis).contiguous()
    except ImportError:  # pragma: no cover
        pass
    Wm = np.moveaxis(np.asarray(W), axis, -1)
    Wm = Wm - (Wm @ Q) @ Q.T
    return np.moveaxis(Wm, -1, axis)


def subspace_overlap(Q_ref: np.ndarray, Q: np.ndarray) -> float:
    """Fraction of span(Q_ref) energy captured by span(Q): ||Q_ref^T Q||_F^2 / dim(Q_ref)."""
    return float(np.sum((Q_ref.T @ Q) ** 2) / Q_ref.shape[1])


def dft_bin_freq(k: int, n: int = PATCH, tr: float = TR) -> float:
    """Frequency (Hz) of DFT bin k on an n-sample window with spacing tr: k / (n * tr)."""
    return k / (n * tr)


def band_basis(kmin: int = 2, n: int = PATCH, tr: float = TR) -> np.ndarray:
    """Orthonormal basis of the DFT bins k >= kmin on n samples (cos and sin; Nyquist has cos only).
    For n = 16, kmin = 2: 6 x 2 + 1 = 13 dims (>= 0.174 Hz at TR 0.72 s)."""
    t = np.arange(n)
    cols = []
    for k in range(kmin, n // 2 + 1):
        cols.append(np.cos(2 * np.pi * k * t / n))
        if k < n // 2:
            cols.append(np.sin(2 * np.pi * k * t / n))
    Q, _ = np.linalg.qr(np.stack(cols, axis=1))
    return Q


def random_in_span(rng: np.random.Generator, B: np.ndarray, k: int) -> np.ndarray:
    """Haar-random k-dim orthonormal subspace inside span(B) (B orthonormal, n x m)."""
    return B @ random_basis(rng, k, n=B.shape[1])


def carrier_share(dW, Qc: np.ndarray, axis: int = KERNEL_TIME_AXIS) -> float:
    """rho = ||dW Qc||^2 / ||dW||^2 along the temporal axis (as scripts/phase2_confirm_analyze.py)."""
    D = np.moveaxis(np.asarray(dW, dtype=np.float64), axis, -1)
    tot = float(np.sum(D ** 2))
    return float(np.sum((D @ Qc) ** 2) / tot) if tot > 0 else 0.0
