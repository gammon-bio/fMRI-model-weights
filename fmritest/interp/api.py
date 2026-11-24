"""Inspection and intervention API for MaskedRegionTransformer.

Hook names (see `Inspector.hook_names()`):
  hook_embed
  blocks.{l}.hook_resid_pre | hook_attn_out | hook_resid_mid | hook_mlp_out | hook_resid_post   (B, R, d)
  blocks.{l}.attn.hook_pattern   (B, H, R, R)  attention probabilities
  blocks.{l}.attn.hook_z         (B, R, H, dh) per-head outputs before W_O
  blocks.{l}.mlp.hook_post       (B, R, d_mlp) MLP neuron activations

Interventions are either *activation hooks* (ablate / patch / remove_direction; live until
restore) or *weight edits* (project_out_weight / set_weight; modify parameters in place).
`restore()` removes every hook, reloads the original state dict, and verifies a SHA-256 hash
of all parameters and buffers against the hash taken at construction. Every intervention and
every restore is appended to a JSONL log with config, seed and a diff summary.
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from ..models.transformer import HookPoint, MaskedRegionTransformer


def state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _unit(v) -> torch.Tensor:
    v = torch.as_tensor(np.asarray(v), dtype=torch.float32).flatten()
    n = v.norm()
    if n == 0:
        raise ValueError("direction has zero norm")
    return v / n


class Inspector:
    def __init__(self, model: MaskedRegionTransformer, log_path: str | Path | None = None,
                 config: dict | None = None, seed: int | None = None, ref_batch=None):
        self.model = model.eval()
        self.log_path = Path(log_path) if log_path else None
        self.config, self.seed = config or {}, seed
        self._orig_state = copy.deepcopy(model.state_dict())
        self._orig_hash = state_hash(model)
        self._handles: list = []
        self._active: list[dict] = []
        self.ref_batch = ref_batch  # optional (x, mask) used to summarise output change per intervention
        self._ref_out = self._forward_ref()

    # ------------------------------------------------------------------ reading
    def hook_names(self) -> list[str]:
        return [n for n, m in self.model.named_modules() if isinstance(m, HookPoint)]

    def _module(self, name: str) -> HookPoint:
        mods = dict(self.model.named_modules())
        if name not in mods or not isinstance(mods[name], HookPoint):
            raise KeyError(f"unknown hook {name!r}; see hook_names()")
        return mods[name]

    def weights(self, name: str | None = None):
        params = dict(self.model.named_parameters())
        return {k: v.detach().clone() for k, v in params.items()} if name is None else params[name].detach().clone()

    def head_weights(self, layer: int, head: int) -> dict:
        """Per-head slices: W_Q/W_K/W_V rows (dh x d), W_O columns (d x dh)."""
        attn = self.model.blocks[layer].attn
        sl = slice(head * attn.dh, (head + 1) * attn.dh)
        return {"W_Q": attn.W_Q.weight[sl].detach().clone(), "W_K": attn.W_K.weight[sl].detach().clone(),
                "W_V": attn.W_V.weight[sl].detach().clone(), "W_O": attn.W_O.weight[:, sl].detach().clone()}

    def neuron_weights(self, layer: int, neuron: int) -> dict:
        mlp = self.model.blocks[layer].mlp
        return {"W_in": mlp.W_in.weight[neuron].detach().clone(), "W_out": mlp.W_out.weight[:, neuron].detach().clone()}

    @torch.no_grad()
    def run_with_cache(self, x, mask=None, names: list[str] | None = None):
        names = names or self.hook_names()
        cache, hs = {}, []
        for n in names:
            hs.append(self._module(n).register_forward_hook(
                lambda m, i, o, n=n: cache.__setitem__(n, o.detach().clone())))
        try:
            out = self.model(torch.as_tensor(x), None if mask is None else torch.as_tensor(mask))
        finally:
            for h in hs:
                h.remove()
        return out, cache

    @torch.no_grad()
    def run_with_hooks(self, x, mask=None, hooks: dict | None = None):
        """Temporary hooks {name: fn(activation) -> activation or None}; removed after the call."""
        hs = [self._module(n).register_forward_hook(lambda m, i, o, f=f: f(o)) for n, f in (hooks or {}).items()]
        try:
            return self.model(torch.as_tensor(x), None if mask is None else torch.as_tensor(mask))
        finally:
            for h in hs:
                h.remove()

    # ------------------------------------------------------------ interventions
    def _add_hook(self, name, fn, record):
        self._handles.append(self._module(name).register_forward_hook(lambda m, i, o: fn(o)))
        self._active.append(record)
        self._log(record, weight_diff=None)

    @staticmethod
    def _select(name, head=None, neuron=None, positions=None):
        """Index tuple for the targeted unit. Region positions index axis 1 (query rows for patterns)."""
        pos = slice(None) if positions is None else list(positions)
        if name.endswith("hook_pattern"):
            if neuron is not None:
                raise ValueError("neuron index invalid for hook_pattern")
            return (slice(None), slice(None) if head is None else head, pos)
        if head is not None:
            if not name.endswith("hook_z"):
                raise ValueError("head index only valid for hook_z / hook_pattern")
            return (slice(None), pos, head)
        if neuron is not None:
            return (slice(None), pos, neuron)
        return (slice(None), pos)

    def ablate(self, name: str, kind: str = "zero", head: int | None = None, neuron: int | None = None,
               mean_value=None, positions=None):
        """Zero- or mean-ablate an activation (optionally one head / neuron / set of region positions)."""
        if kind == "mean" and mean_value is None:
            raise ValueError("mean ablation requires mean_value (computed on data the agent may use)")
        mv = None if mean_value is None else torch.as_tensor(np.asarray(mean_value), dtype=torch.float32)

        idx = self._select(name, head, neuron, positions)

        def fn(o):
            o = o.clone()
            o[idx] = 0.0 if mv is None else mv
            return o

        self._add_hook(name, fn, {"type": "ablate", "hook": name, "kind": kind, "head": head,
                                  "neuron": neuron, "positions": positions})

    def patch(self, name: str, source_activation, head: int | None = None, neuron: int | None = None,
              positions=None):
        """Replace an activation (or one head/neuron of it) with a cached activation from another run.
        The source must have the same batch shape as the inputs it will be applied to."""
        src = torch.as_tensor(source_activation)
        idx = self._select(name, head, neuron, positions)

        def fn(o):
            if o.shape != src.shape:
                raise ValueError(f"patch shape mismatch {tuple(o.shape)} vs {tuple(src.shape)}")
            o = o.clone()
            o[idx] = src[idx]
            return o

        self._add_hook(name, fn, {"type": "patch", "hook": name, "head": head, "neuron": neuron,
                                  "positions": positions, "source_norm": float(src.norm())})

    def remove_direction(self, name: str, direction, head: int | None = None):
        """Project a unit direction out of the last axis of an activation: a <- a - (a.v) v."""
        v = _unit(direction)
        if head is not None and not name.endswith("hook_z"):
            raise ValueError("head index only valid for hook_z in remove_direction")

        def fn(o):
            o = o.clone()
            if head is not None:
                a = o[:, :, head]
                o[:, :, head] = a - (a @ v)[..., None] * v
            else:
                o = o - (o @ v)[..., None] * v
            return o

        self._add_hook(name, fn, {"type": "remove_direction", "hook": name, "head": head,
                                  "direction": v.tolist()})

    def apply_affine(self, name: str, fn_np, label: str):
        """Apply an arbitrary fitted map (e.g. a LEACE eraser) to an activation's last axis."""
        def fn(o):
            sh = o.shape
            y = fn_np(o.reshape(-1, sh[-1]).double().numpy())
            return torch.as_tensor(y, dtype=o.dtype).reshape(sh)

        self._add_hook(name, fn, {"type": "affine", "hook": name, "label": label})

    @torch.no_grad()
    def project_out_weight(self, param: str, direction, side: str = "out"):
        """Rank-1 weight edit. side='out': W <- (I - v v^T) W (removes v from the layer's output space;
        bias is projected too). side='in': W <- W (I - v v^T) (layer becomes blind to input direction v)."""
        params = dict(self.model.named_parameters())
        W = params[param]
        v = _unit(direction)
        before = W.detach().clone()
        if side == "out":
            W.sub_(torch.outer(v, v @ W))
            bname = param.replace("weight", "bias")
            if bname != param and bname in params:
                b = params[bname]
                b.sub_(v * (v @ b))
        elif side == "in":
            W.sub_(torch.outer(W @ v, v))
        else:
            raise ValueError(side)
        rec = {"type": "project_out_weight", "param": param, "side": side, "direction": v.tolist()}
        self._active.append(rec)
        self._log(rec, weight_diff={param: before})

    @torch.no_grad()
    def set_weight(self, param: str, value, note: str = ""):
        W = dict(self.model.named_parameters())[param]
        before = W.detach().clone()
        W.copy_(torch.as_tensor(np.asarray(value), dtype=W.dtype))
        rec = {"type": "set_weight", "param": param, "note": note}
        self._active.append(rec)
        self._log(rec, weight_diff={param: before})

    # ------------------------------------------------------------------ restore
    def restore(self) -> str:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.model.load_state_dict(self._orig_state)
        h = state_hash(self.model)
        if h != self._orig_hash:
            raise RuntimeError("restore failed: parameter hash differs from original")
        n = len(self._active)
        self._active.clear()
        self._log({"type": "restore", "n_interventions_undone": n}, weight_diff=None)
        return h

    @contextmanager
    def intervention(self):
        """`with insp.intervention(): ...` -- everything applied inside is undone on exit, even on error."""
        try:
            yield self
        finally:
            self.restore()

    @property
    def active(self) -> list[dict]:
        return list(self._active)

    def is_pristine(self) -> bool:
        return not self._handles and state_hash(self.model) == self._orig_hash

    # ------------------------------------------------------------------ logging
    @torch.no_grad()
    def _forward_ref(self):
        if self.ref_batch is None:
            return None
        x, m = self.ref_batch
        return self.model(torch.as_tensor(x), torch.as_tensor(m)).clone()

    def _log(self, record: dict, weight_diff: dict | None):
        diff = {}
        if weight_diff:
            params = dict(self.model.named_parameters())
            for k, before in weight_diff.items():
                d = params[k].detach() - before
                diff[k] = {"fro_delta": float(d.norm()), "rel_delta": float(d.norm() / before.norm().clamp_min(1e-12)),
                           "n_changed": int((d.abs() > 0).sum()), "numel": int(d.numel())}
        if self.ref_batch is not None and record["type"] != "restore":
            out = self._forward_ref()
            diff["ref_output_mse_change"] = float(((out - self._ref_out) ** 2).mean())
        entry = {"time": _dt.datetime.now(_dt.timezone.utc).isoformat(), "seed": self.seed,
                 "config": self.config, "intervention": {k: v for k, v in record.items() if k != "direction"},
                 "direction_dim": len(record["direction"]) if "direction" in record else None,
                 "n_active": len(self._active), "param_hash": state_hash(self.model),
                 "original_hash": self._orig_hash, "diff": diff}
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        return entry
