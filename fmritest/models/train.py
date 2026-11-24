"""Training loop and windowing utilities for the masked region transformer."""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .transformer import MaskedRegionTransformer, build_model, masked_mse


def set_seed(seed: int, n_threads: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(n_threads)
    torch.use_deterministic_algorithms(True)


def to_windows(X: np.ndarray, window: int) -> np.ndarray:
    """Non-overlapping windows. X: (N, T, R) -> (N, n_win, R, W); trailing remainder dropped."""
    N, T, R = X.shape
    n_win = T // window
    Xw = X[:, : n_win * window].reshape(N, n_win, window, R)
    return Xw.transpose(0, 1, 3, 2).copy()


def random_masks(rng: np.random.Generator, n: int, n_regions: int, mask_ratio: float) -> np.ndarray:
    n_mask = max(1, int(round(mask_ratio * n_regions)))
    m = np.zeros((n, n_regions), dtype=bool)
    for i in range(n):
        m[i, rng.choice(n_regions, n_mask, replace=False)] = True
    return m


def make_eval_set(X: np.ndarray, window: int, masks_per_window: int, mask_ratio: float, seed: int):
    """Fixed evaluation windows and masks (shared verbatim by the model and all baselines).

    Returns x: (M, R, W), mask: (M, R) bool, subj: (M,) index into X's first axis.
    """
    Xw = to_windows(X, window)  # (N, nw, R, W)
    N, nw, R, W = Xw.shape
    rng = np.random.default_rng(seed + 104729)
    x = np.repeat(Xw.reshape(N * nw, R, W), masks_per_window, axis=0)
    subj = np.repeat(np.repeat(np.arange(N), nw), masks_per_window)
    mask = random_masks(rng, x.shape[0], R, mask_ratio)
    return x.astype(np.float32), mask, subj


def sample_batch(rng, X: np.ndarray, batch: int, window: int, mask_ratio: float):
    N, T, R = X.shape
    s = rng.integers(0, N, batch)
    t0 = rng.integers(0, T - window + 1, batch)
    x = np.stack([X[i, t:t + window].T for i, t in zip(s, t0)]).astype(np.float32)
    return x, random_masks(rng, batch, R, mask_ratio)


@torch.no_grad()
def eval_mse(model, x: np.ndarray, mask: np.ndarray, batch: int = 512) -> float:
    model.eval()
    tot, cnt = 0.0, 0.0
    for i in range(0, len(x), batch):
        xb, mb = torch.from_numpy(x[i:i + batch]), torch.from_numpy(mask[i:i + batch])
        pred = model(xb, mb)
        tot += float(((pred - xb) ** 2 * mb[..., None]).sum())
        cnt += float(mb.sum()) * xb.shape[-1]
    return tot / cnt


def train_model(X_train: np.ndarray, val_set, cfg: dict, seed: int, log=print):
    """Train from scratch. Returns (model, curves). Final weights are the last step (no selection
    on validation or test data)."""
    mc, tc = cfg["model"], cfg["train"]
    set_seed(seed, cfg["compute"]["n_threads"])
    model = build_model(mc, X_train.shape[2])
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])

    def lr_at(step):
        if step < tc["warmup"]:
            return (step + 1) / tc["warmup"]
        p = (step - tc["warmup"]) / max(1, tc["steps"] - tc["warmup"])
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = np.random.default_rng(seed + 31337)
    curves = {"step": [], "train": [], "val": []}
    run, n_run, t0 = 0.0, 0, time.time()
    for step in range(tc["steps"]):
        model.train()
        x, m = sample_batch(rng, X_train, tc["batch_size"], mc["window"], mc["mask_ratio"])
        x, m = torch.from_numpy(x), torch.from_numpy(m)
        loss = masked_mse(model(x, m), x, m)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run += float(loss.detach())
        n_run += 1
        if (step + 1) % tc["eval_every"] == 0 or step == tc["steps"] - 1:
            v = eval_mse(model, *val_set[:2])
            curves["step"].append(step + 1)
            curves["train"].append(run / n_run)
            curves["val"].append(v)
            run, n_run = 0.0, 0
            if (step + 1) % (tc["eval_every"] * 10) == 0:
                log(f"  step {step + 1}: train {curves['train'][-1]:.4f} val {v:.4f} ({time.time() - t0:.0f}s)")
    curves["seconds"] = time.time() - t0
    model.eval()
    return model, curves


def untrained_model(cfg: dict, n_regions: int, seed: int, init: int = 0) -> MaskedRegionTransformer:
    """Same architecture at a random initialisation (control for 'structure inherited from input').
    `init` indexes independent random inits for building an untrained-model null distribution."""
    set_seed(seed * 1000 + 999 + init, cfg["compute"]["n_threads"])
    return build_model(cfg["model"], n_regions).eval()
