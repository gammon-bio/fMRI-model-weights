"""Config loading and thread capping."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "configs" / "default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cap_threads(n: int) -> None:
    """Cap BLAS/OpenMP threads. Must run before numpy/torch spin up their pools to fully apply."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(n)
    import torch
    torch.set_num_threads(n)


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Return a deep copy of cfg with dotted-key overrides applied (e.g. {"data.confound.strength": 0})."""
    import copy
    cfg = copy.deepcopy(cfg)
    for key, val in (overrides or {}).items():
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d[p]
        if parts[-1] not in d:
            raise KeyError(f"override {key!r} does not match an existing config key")
        d[parts[-1]] = val
    return cfg
