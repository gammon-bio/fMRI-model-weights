"""BrainLM vitmae_111M encoder (no masking), layer-9 cache, top-2-layer diagnosis head, kernel edits.

- The encoder is stock HF `ViTMAEModel`. Masking is bypassed entirely (equivalent to mask_ratio 0 with
  an identity shuffle): patch embedding + fixed sin-cos position embedding + CLS, then the layers.
  `check_against_hf` verifies this equals `ViTMAEModel.forward` with mask_ratio 0 and noise = arange.
- Layers 0..9 are frozen; their output (input to layer 10) is cached as fp16 once per scan.
- `HeadModel` = trainable copies of layers 10-11 + the final layernorm + a linear head on the mean of
  the real-data patch tokens (CLS and padded patches excluded).
- `edited_kernel` temporarily replaces the patch-embedding kernel for a full forward.
"""
from __future__ import annotations

import copy
import hashlib
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

WEIGHTS_SHA256 = "25f5b52178f7409dee4fed2fa3dd18c6692d0764d96996ff0ee4be740520fcd6"
SPLIT_LAYER = 10  # cache = input to this layer; layers SPLIT_LAYER.. are fine-tuned


def sha256(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_encoder(weights_dir: str | Path, attn: str = "sdpa", verify: bool = True):
    """Load the vitmae_111M encoder (decoder discarded), eval mode, mask_ratio 0, frozen."""
    from transformers import ViTMAEConfig, ViTMAEForPreTraining
    d = Path(weights_dir)
    if verify and sha256(d / "pytorch_model.bin") != WEIGHTS_SHA256:
        raise RuntimeError("vitmae_111M weights sha256 mismatch")
    cfg = ViTMAEConfig.from_pretrained(d)
    cfg.mask_ratio = 0.0
    cfg._attn_implementation = attn
    full = ViTMAEForPreTraining(cfg)
    sd = torch.load(d / "pytorch_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = full.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    enc = full.vit.eval()
    del full, sd
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def embed(enc, pixels: torch.Tensor) -> torch.Tensor:
    """(B, 3, 432, 432) -> (B, 1 + 729, 768): CLS + patch tokens in raster order, positions added."""
    e = enc.embeddings
    pos = e.position_embeddings
    x = e.patch_embeddings(pixels) + pos[:, 1:, :]
    cls = (e.cls_token + pos[:, :1, :]).expand(x.shape[0], -1, -1)
    return torch.cat([cls, x], dim=1)


def _amp(enabled: bool):
    """bf16 autocast on CPU (LayerNorm / softmax stay fp32 under autocast)."""
    return torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled)


@torch.no_grad()
def encode_to_layer(enc, pixels: np.ndarray, upto: int = SPLIT_LAYER, batch_size: int = 8,
                    out_dtype=torch.float16, amp: bool = False) -> torch.Tensor:
    """Hidden states entering layer `upto` (i.e. after layers 0..upto-1), (N, 730, 768) in out_dtype."""
    outs = []
    with _amp(amp):
        for i in range(0, len(pixels), batch_size):
            h = embed(enc, torch.from_numpy(np.ascontiguousarray(pixels[i:i + batch_size])).float())
            for layer in enc.encoder.layer[:upto]:
                h = layer(h)[0]
            outs.append(h.float().to(out_dtype))
    return torch.cat(outs)


def encode_recordings(enc, Xs: np.ndarray, batch_size: int = 8, chunk: int = 32, amp: bool = False,
                      out_dtype=torch.float16) -> torch.Tensor:
    """Robust-scaled recordings (N, T, R) -> layer-10 input (N, 730, 768); packs chunk-wise to save RAM."""
    from .pack import pack
    return torch.cat([encode_to_layer(enc, pack(Xs[i:i + chunk]), batch_size=batch_size, amp=amp,
                                      out_dtype=out_dtype) for i in range(0, len(Xs), chunk)])


@torch.no_grad()
def check_against_hf(enc, pixels: np.ndarray) -> float:
    """Max abs difference between our final hidden state and ViTMAEModel.forward(mask_ratio=0, noise=arange)."""
    x = torch.from_numpy(pixels).float()
    noise = torch.arange(729, dtype=torch.float32).expand(len(x), -1)
    ref = enc(pixel_values=x, noise=noise).last_hidden_state
    h = encode_to_layer(enc, pixels, upto=len(enc.encoder.layer), out_dtype=torch.float32)
    ours = enc.layernorm(h)
    return float((ours - ref).abs().max())


@contextmanager
def edited_kernel(enc, W_new: torch.Tensor):
    """Temporarily replace the patch-embedding kernel (restored on exit, even on error)."""
    w = enc.embeddings.patch_embeddings.projection.weight
    old = w.data.clone()
    try:
        w.data.copy_(W_new)
        yield enc
    finally:
        w.data.copy_(old)


def kernel(enc) -> torch.Tensor:
    return enc.embeddings.patch_embeddings.projection.weight.detach().clone()


class HeadModel(nn.Module):
    """Copies of encoder layers SPLIT_LAYER.. + final layernorm + linear head on real-data token mean."""

    def __init__(self, enc, token_idx: np.ndarray, seed: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for layer in enc.encoder.layer[SPLIT_LAYER:]])
        self.ln = copy.deepcopy(enc.layernorm)
        for p in self.parameters():
            p.requires_grad_(True)
        g = torch.Generator().manual_seed(seed)
        self.head = nn.Linear(enc.config.hidden_size, 1)
        with torch.no_grad():
            self.head.weight.normal_(0.0, 0.02, generator=g)
            self.head.bias.zero_()
        self.register_buffer("token_idx", torch.as_tensor(token_idx, dtype=torch.long))

    def features(self, h: torch.Tensor) -> torch.Tensor:
        h = h.float()
        for layer in self.layers:
            h = layer(h)[0]
        return self.ln(h)[:, self.token_idx].mean(1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(h)).squeeze(-1)


def token_index(patch_mask: np.ndarray) -> np.ndarray:
    """Sequence indices (CLS = 0) of the real-data patch tokens."""
    return np.flatnonzero(patch_mask.ravel()) + 1


@torch.no_grad()
def predict(model: HeadModel, H: torch.Tensor, batch_size: int = 32, feats: bool = False,
            amp: bool = False) -> np.ndarray:
    model.eval()
    fn = model.features if feats else model
    with _amp(amp):
        return torch.cat([fn(H[i:i + batch_size]).float() for i in range(0, len(H), batch_size)]).numpy()


def _wloss(model, H, y, w, amp, batch_size=32):
    logits = torch.from_numpy(predict(model, H, batch_size, amp=amp))
    return float((F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / w.sum())


def group_balanced_order(rng: np.random.Generator, group_ids: np.ndarray) -> np.ndarray:
    """len(group_ids) indices: each draw picks a group uniformly, then a member uniformly (with replacement)."""
    groups = [np.flatnonzero(group_ids == g) for g in np.unique(group_ids)]
    g = rng.integers(0, len(groups), len(group_ids))
    return np.array([groups[k][rng.integers(0, len(groups[k]))] for k in g], dtype=np.int64)


def train_head(enc, H_train: torch.Tensor, y_train: np.ndarray, w_train: np.ndarray,
               H_val: torch.Tensor, y_val: np.ndarray, w_val: np.ndarray, token_idx: np.ndarray,
               cfg: dict, seed: int, amp: bool = False, log=print, group_ids: np.ndarray | None = None):
    """Fine-tune layers 10-11 + ln + head with weighted BCE (AdamW).
    cfg keys: epochs, batch_size, lr_head, lr_layers, weight_decay, patience, and optional
    early_stopping (default True = Stage 0: keep the best weighted-val-loss weights, stop after
    `patience` bad epochs; False = Stage 0b C3: run all epochs and keep the FINAL weights).
    group_ids (Stage 0b C2): if given, each epoch draws len(train) samples by group-balanced sampling
    with replacement (a group uniformly, then a member uniformly); pass w_train = ones then."""
    torch.manual_seed(seed)
    model = HeadModel(enc, token_idx, seed + 1)
    head_ids = {id(p) for p in model.head.parameters()}
    opt = torch.optim.AdamW([
        {"params": list(model.head.parameters()), "lr": cfg["lr_head"]},
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": cfg["lr_layers"]},
    ], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(seed + 2)
    yt = torch.from_numpy(y_train.astype(np.float32))
    wt = torch.from_numpy(w_train.astype(np.float32))
    yv = torch.from_numpy(y_val.astype(np.float32))
    wv = torch.from_numpy(w_val.astype(np.float32))
    best, best_state, best_ep, bad, curve = np.inf, None, -1, 0, []
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(len(H_train)) if group_ids is None else group_balanced_order(rng, group_ids)
        run = 0.0
        for i in range(0, len(order), cfg["batch_size"]):
            b = torch.from_numpy(order[i:i + cfg["batch_size"]])
            with _amp(amp):
                logit = model(H_train[b]).float()
            loss = (F.binary_cross_entropy_with_logits(logit, yt[b], reduction="none") * wt[b]).sum() / wt[b].sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss.detach()) * len(b)
        fixed = not cfg.get("early_stopping", True)
        last = ep + 1 == cfg["epochs"]
        vl = _wloss(model, H_val, yv, wv, amp) if (not fixed or last) else float("nan")
        curve.append({"epoch": ep + 1, "train_loss": run / len(order), "val_loss": vl, "t": time.time() - t0})
        log(f"    epoch {ep + 1}: train {run / len(order):.4f} val {vl:.4f} ({time.time() - t0:.0f}s)")
        if not cfg.get("early_stopping", True):
            best, best_ep = vl, ep + 1
            continue
        if vl < best - 1e-4:
            best, best_ep, bad = vl, ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if cfg.get("early_stopping", True):
        model.load_state_dict(best_state)
    model.eval()
    return model, {"curve": curve, "best_epoch": best_ep, "best_val_loss": best, "seconds": time.time() - t0}
