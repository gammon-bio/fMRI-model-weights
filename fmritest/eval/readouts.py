"""Readouts of model internals used by the evaluation (all go through the interp API hooks)."""
from __future__ import annotations

import numpy as np
import torch

from ..interp.api import Inspector


@torch.no_grad()
def masked_query_attention(insp: Inspector, xw: np.ndarray, layers=None, batch_windows: int = 16) -> np.ndarray:
    """Pre-registered primary network readout.

    For every window and every region i, mask region i alone and record the attention row of
    query token i (which carries no data, only region identity + mask embedding). A[i, j] is the
    attention from masked region i to region j, averaged over windows, heads and the chosen
    layers. The diagonal is zeroed and A is symmetrised by the extractor.
    xw: (n_windows, R, W).
    """
    model = insp.model
    R = xw.shape[1]
    L = len(model.blocks)
    layers = list(range(L)) if layers is None else layers
    names = [f"blocks.{l}.attn.hook_pattern" for l in layers]
    eye = torch.eye(R, dtype=torch.bool)
    A = torch.zeros(R, R)
    n = 0
    for i in range(0, len(xw), batch_windows):
        xb = torch.from_numpy(xw[i:i + batch_windows]).float()
        b = xb.shape[0]
        x_rep = xb.repeat_interleave(R, dim=0)                      # (b*R, R, W)
        m_rep = eye.repeat(b, 1)                                    # sample k masks region k % R
        _, cache = insp.run_with_cache(x_rep, m_rep, names)
        for nm in names:
            p = cache[nm].mean(1).view(b, R, R, R)                   # (b, masked i, query, key)
            A += p[:, torch.arange(R), torch.arange(R), :].sum(0)    # row i of query i when i masked
        n += b * len(names)
    A = (A / n).numpy()
    np.fill_diagonal(A, 0.0)
    return A


def region_embedding_similarity(insp: Inspector) -> np.ndarray:
    """Secondary readout (data-free): cosine similarity of learned region identity embeddings."""
    E = insp.weights("region_emb").numpy()
    E = E - E.mean(0)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    np.fill_diagonal(S, 0.0)
    return S


@torch.no_grad()
def window_embeddings(insp: Inspector, xw: np.ndarray, hook: str | None = None, pool: bool = True,
                      batch: int = 256, masks: np.ndarray | None = None) -> np.ndarray:
    """Final-layer residual stream (unmasked windows unless `masks` given). pool=True: mean over
    region tokens -> (n, d); pool=False: per-token -> (n, R, d)."""
    hook = hook or f"blocks.{len(insp.model.blocks) - 1}.hook_resid_post"
    out = []
    for i in range(0, len(xw), batch):
        mb = None if masks is None else torch.from_numpy(masks[i:i + batch])
        _, cache = insp.run_with_cache(torch.from_numpy(xw[i:i + batch]).float(), mb, [hook])
        h = cache[hook]
        out.append((h.mean(1) if pool else h).numpy())
    return np.concatenate(out)
