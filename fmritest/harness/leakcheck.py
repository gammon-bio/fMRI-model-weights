"""Raw-input leak check: is a served activation an affine function of the raw input window?

For every candidate hook, random 48-TR crops of training subjects are run through the model.
Token-level hooks (one row per region token) are regressed on [that token's raw time series,
one-hot region id]; window-level hooks (`head_in`) on the whole flattened window plus a constant.
R^2 is measured on held-out crops (fit and test crops come from disjoint subjects), so an
over-parameterised fit cannot pass for a leak. A hook with held-out R^2 above the threshold is
refused. `hook_embed` and `blocks.0.hook_resid_pre` are exactly affine (the latter is the
former's output) and are refused in configs/harness.yaml regardless.
"""
from __future__ import annotations

import numpy as np
import torch

from .modelio import HEAD_IN, _run_windows


def _crops(Xs: np.ndarray, W: int, n: int, rng) -> np.ndarray:
    s = rng.integers(0, len(Xs), n)
    t = rng.integers(0, Xs.shape[1] - W + 1, n)
    return np.stack([Xs[i, tt:tt + W].T for i, tt in zip(s, t)]).astype(np.float32)   # (n, R, W)


def _token_rows(name: str, a: torch.Tensor) -> np.ndarray:
    """(B, ...) activation -> (B, R, D) rows per region token."""
    a = a.numpy().astype(np.float64)
    if name.endswith("hook_pattern"):                       # (B, H, R, R): query token rows
        a = a.transpose(0, 2, 1, 3)
    return a.reshape(a.shape[0], a.shape[1], -1)


def _r2(A_fit, Y_fit, A_te, Y_te) -> float:
    coef, *_ = np.linalg.lstsq(A_fit, Y_fit, rcond=None)
    res = ((Y_te - A_te @ coef) ** 2).sum()
    tot = ((Y_te - Y_te.mean(0)) ** 2).sum()
    return float(1.0 - res / max(tot, 1e-300))


@torch.no_grad()
def affine_leak_r2(model, Xs: np.ndarray, hooks: list[str], seed: int = 0, n_token_crops: int = 200,
                   n_window_crops: int = 4000, inverse: bool = False) -> dict[str, float]:
    """Held-out R^2 of activation ~ affine(input) (inverse=False) or input ~ affine(activation)
    (inverse=True, token-level hooks only: how much of the raw window an agent could decode)."""
    rng = np.random.default_rng(seed)
    half = len(Xs) // 2
    W, R = model.window, model.n_regions
    out = {}
    tok = [h for h in hooks if h != HEAD_IN]
    if tok:
        xf, xt = _crops(Xs[:half], W, n_token_crops, rng), _crops(Xs[half:], W, n_token_crops, rng)
        _, cf = _run_windows(model, torch.from_numpy(xf), tok)
        _, ct = _run_windows(model, torch.from_numpy(xt), tok)
        eye = np.eye(R)

        def design(x):
            n = x.shape[0]
            return np.concatenate([x.reshape(n * R, W), np.tile(eye, (n, 1))], 1).astype(np.float64)
        Af, At = design(xf), design(xt)
        for h in tok:
            Yf, Yt = _token_rows(h, cf[h]), _token_rows(h, ct[h])
            Yf, Yt = Yf.reshape(-1, Yf.shape[-1]), Yt.reshape(-1, Yt.shape[-1])
            if inverse:
                Bf = np.concatenate([Yf, np.tile(eye, (len(xf), 1))], 1)
                Bt = np.concatenate([Yt, np.tile(eye, (len(xt), 1))], 1)
                out[h] = _r2(Bf, xf.reshape(-1, W).astype(np.float64), Bt, xt.reshape(-1, W).astype(np.float64))
            else:
                out[h] = _r2(Af, Yf, At, Yt)
    if HEAD_IN in hooks and not inverse:
        n = n_window_crops
        xf, xt = _crops(Xs[:half], W, n, rng), _crops(Xs[half:], W, n // 4, rng)
        _, cf = _run_windows(model, torch.from_numpy(xf), [HEAD_IN])
        _, ct = _run_windows(model, torch.from_numpy(xt), [HEAD_IN])
        design = lambda x: np.concatenate([x.reshape(len(x), -1), np.ones((len(x), 1))], 1).astype(np.float64)
        out[HEAD_IN] = _r2(design(xf), cf[HEAD_IN].numpy().astype(np.float64),
                           design(xt), ct[HEAD_IN].numpy().astype(np.float64))
    return out


def leak_report(model, Xs: np.ndarray, hooks: list[str], threshold: float, seed: int = 0, fast: bool = False) -> dict:
    kw = {"n_token_crops": 60, "n_window_crops": 2000} if fast else {}
    fwd = affine_leak_r2(model, Xs, hooks, seed, **kw)
    inv = affine_leak_r2(model, Xs, hooks, seed, inverse=True, **kw)
    flagged = sorted(h for h in hooks if fwd.get(h, 0) > threshold or inv.get(h, 0) > threshold)
    return {"threshold": threshold, "forward_r2": fwd, "inverse_r2": inv, "flagged": flagged}
