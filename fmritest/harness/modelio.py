"""Model loading, weight files, edit validation / application / diff, and deterministic readouts.

Readouts:
- `oracle_readout` computes each subject on its own (batch = that subject's windows), so a subject's
  numbers do not depend on which other subjects share a call. That is what lets a scripted agent
  that collects activations over several oracle calls reproduce a harness comparator bit for bit.
- `SetCache` / `set_logits` score a fixed sealed set in one batch and can restart the forward pass
  at the first edited stage (same modules, same inputs -> bitwise identical to a full pass;
  tested), which makes the 200-draw nulls cheap for late-layer edits.
"""
from __future__ import annotations

import copy
import hashlib
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from ..finetune import DiagnosisModel
from ..models.train import to_windows
from ..models.transformer import HookPoint, build_model

HEAD_IN = "head_in"      # pseudo-hook: mean over regions of ln_final(resid_post of the last block)
LOGIT = "logit"


class EditError(ValueError):
    """An edit file that cannot be applied (unknown tensor, wrong shape, non-finite, unreadable)."""


def build_diag_model(cfg_model: dict, n_regions: int) -> DiagnosisModel:
    return DiagnosisModel.from_backbone(build_model(cfg_model, n_regions)).eval()


def load_diag_model(cfg_model: dict, n_regions: int, path: str | Path) -> DiagnosisModel:
    m = build_diag_model(cfg_model, n_regions)
    m.load_state_dict(torch.load(path, weights_only=True))
    return m.eval()


def numpy_weights(model: torch.nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().astype(np.float32).copy() for k, v in model.state_dict().items()}


def save_weights_npz(path: str | Path, weights: dict[str, np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez(f, **weights)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hook_names(model: torch.nn.Module) -> list[str]:
    return [n for n, m in model.named_modules() if isinstance(m, HookPoint)]


def served_hooks(model: torch.nn.Module, refused=("hook_embed",)) -> list[str]:
    return [h for h in hook_names(model) if h not in set(refused)] + [HEAD_IN]


# ----------------------------------------------------------------------------------- edits
def load_edit(path: str | Path, model: torch.nn.Module) -> dict[str, np.ndarray]:
    """Read an edit .npz (tensor name -> full replacement array) and validate it against `model`."""
    try:
        with np.load(path, allow_pickle=False) as z:
            d = {k: np.asarray(z[k]) for k in z.files}
    except Exception as e:  # noqa: BLE001 -- any unreadable file is a refused edit
        raise EditError(f"cannot read edit file {path}: {e}") from None
    validate_edit(d, model)
    return {k: v.astype(np.float32) for k, v in d.items()}


def validate_edit(d: dict, model: torch.nn.Module) -> None:
    sd = model.state_dict()
    if not d:
        raise EditError("edit contains no tensors")
    for k, v in d.items():
        if k not in sd:
            raise EditError(f"unknown tensor {k!r}; valid names are listed in ARCH.md")
        if tuple(v.shape) != tuple(sd[k].shape):
            raise EditError(f"tensor {k!r}: shape {tuple(v.shape)} != required {tuple(sd[k].shape)}")
        if v.dtype.kind not in "fiu":
            raise EditError(f"tensor {k!r}: dtype {v.dtype} is not real numeric")
        if not np.all(np.isfinite(v)):
            raise EditError(f"tensor {k!r}: contains non-finite values")


def apply_edit(model: torch.nn.Module, edit: dict[str, np.ndarray]) -> torch.nn.Module:
    """Fresh copy of `model` with the replacement tensors loaded. `model` is not modified."""
    m = copy.deepcopy(model)
    sd = m.state_dict()
    for k, v in edit.items():
        sd[k] = torch.as_tensor(np.asarray(v, dtype=np.float32)).clone()
    m.load_state_dict(sd)
    return m.eval()


@contextmanager
def edited_in_place(model: torch.nn.Module, edit: dict[str, np.ndarray]):
    """Temporarily load replacement tensors into `model` (restored on exit, even on error)."""
    params = model.state_dict()
    saved = {k: params[k].detach().clone() for k in edit}
    try:
        with torch.no_grad():
            for k, v in edit.items():
                params[k].copy_(torch.as_tensor(np.asarray(v, dtype=np.float32)))
        yield model
    finally:
        with torch.no_grad():
            for k, v in saved.items():
                params[k].copy_(v)


def numerical_rank(delta: np.ndarray, rtol: float) -> tuple[int, np.ndarray]:
    """Rank and singular values of a delta (1-D tensors: rank 1 if non-zero)."""
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim <= 1:
        n = float(np.linalg.norm(d))
        return (1 if n > 0 else 0), np.array([n])
    s = np.linalg.svd(d.reshape(d.shape[0], -1), compute_uv=False)
    if s.size == 0 or s.max() == 0:
        return 0, s
    return int((s > rtol * s.max()).sum()), s


def diff_summary(model: torch.nn.Module, edit: dict[str, np.ndarray], rtol: float = 1e-5) -> dict:
    sd = model.state_dict()
    per, n_changed, fro2 = {}, 0, 0.0
    for k, v in sorted(edit.items()):
        orig = sd[k].detach().cpu().numpy().astype(np.float64)
        d = np.asarray(v, dtype=np.float64) - orig
        nz = int((d != 0).sum())
        rank, s = numerical_rank(d, rtol)
        fro = float(np.linalg.norm(d))
        per[k] = {"shape": list(d.shape), "n_changed": nz, "numel": int(d.size), "fro_delta": fro,
                  "rel_fro_delta": fro / max(float(np.linalg.norm(orig)), 1e-12), "rank": rank,
                  "singular_values": [float(x) for x in s[:max(rank, 1)]]}
        n_changed += nz
        fro2 += fro**2
    changed = [k for k, r in per.items() if r["n_changed"] > 0]
    return {"tensors_in_file": sorted(edit), "tensors_changed": changed, "n_weights_changed": n_changed,
            "fro_delta_total": float(np.sqrt(fro2)), "total_rank": int(sum(per[k]["rank"] for k in changed)),
            "per_tensor": per}


# --------------------------------------------------------------------------------- readouts
@torch.no_grad()
def _run_windows(model: DiagnosisModel, xw: torch.Tensor, hooks: list[str]):
    """One forward pass on windows (B, R, W). Returns window logits (B,) and {hook: (B, ...)}."""
    mods = dict(model.named_modules())
    cache, hs = {}, []
    for n in hooks:
        if n == HEAD_IN:
            continue
        if n not in mods or not isinstance(mods[n], HookPoint):
            raise KeyError(f"unknown hook {n!r}")
        hs.append(mods[n].register_forward_hook(lambda m, i, o, n=n: cache.__setitem__(n, o.detach().clone())))
    try:
        e = model.embed_subject(xw)
        logit = model.head(e).squeeze(-1)
    finally:
        for h in hs:
            h.remove()
    if HEAD_IN in hooks:
        cache[HEAD_IN] = e
    return logit, cache


@torch.no_grad()
def oracle_readout(model: DiagnosisModel, Xs: np.ndarray, hooks: list[str]) -> dict[str, np.ndarray]:
    """Subject-averaged (over non-overlapping windows) activations + subject logit, one subject at a time.
    Xs: (n, T, R) z-scored subjects. Returns {'logit': (n,), hook: (n, ...)} as float32."""
    W = model.window
    out = {LOGIT: [], **{h: [] for h in hooks}}
    for i in range(len(Xs)):
        xw = torch.from_numpy(to_windows(Xs[i:i + 1], W)[0].astype(np.float32))  # (nw, R, W)
        logit, cache = _run_windows(model, xw, hooks)
        out[LOGIT].append(logit.numpy().mean(0))
        for h in hooks:
            out[h].append(cache[h].numpy().mean(0))
    return {k: np.stack(v).astype(np.float32) if v else np.zeros((0,), np.float32) for k, v in out.items()}


# ------------------------------------------------------------------- staged set scoring
def first_stage(names, n_layers: int) -> int:
    """Earliest forward stage an edit can affect. 0: from the input; l+1: input of block l;
    L+1: input of ln_final; L+2: head input (only head / unused tensors edited)."""
    st = n_layers + 2
    for k in names:
        if k.startswith(("embed.", "region_emb", "mask_emb")):
            st = min(st, 0)
        elif k.startswith("blocks."):
            st = min(st, int(re.match(r"blocks\.(\d+)\.", k).group(1)) + 1)
        elif k.startswith("ln_final."):
            st = min(st, n_layers + 1)
    return st


class SetCache:
    """Windows of a fixed subject set plus the unedited model's intermediate states at every stage."""

    @torch.no_grad()
    def __init__(self, model: DiagnosisModel, Xs: np.ndarray):
        xw = to_windows(Xs, model.window).astype(np.float32)
        self.n, self.nw = xw.shape[0], xw.shape[1]
        self.x = torch.from_numpy(xw.reshape(self.n * self.nw, *xw.shape[2:]))
        h = model.hook_embed(model.embed(self.x) + model.region_emb)
        self.stages = {1: h}
        for l, blk in enumerate(model.blocks):
            h = blk(h)
            self.stages[l + 2] = h
        self.stages[len(model.blocks) + 2] = model.ln_final(h).mean(1)


@torch.no_grad()
def set_logits(model: DiagnosisModel, cache: SetCache, stage: int = 0) -> np.ndarray:
    """Subject logits on the cached set, recomputing from `stage` with `model`'s (edited) weights."""
    L = len(model.blocks)
    if stage >= L + 2:
        e = cache.stages[L + 2]
    else:
        if stage <= 0:
            h, start = model.hook_embed(model.embed(cache.x) + model.region_emb), 1
        else:
            h, start = cache.stages[stage], stage
        for l in range(start - 1, L):
            h = model.blocks[l](h)
        e = model.ln_final(h).mean(1)
    return model.head(e).squeeze(-1).numpy().reshape(cache.n, cache.nw).mean(1)
