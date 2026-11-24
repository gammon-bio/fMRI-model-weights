"""One-pass calibration of the `hard` condition: FC eig+varimax network recovery ~0.75.

Uses exactly the pipeline's training split (training subjects only), over calibration seeds
100-103 so the held-out site rotates through all 4 sites. A single grid is evaluated once; the
setting closest to the target is taken if it lies within +/-0.03, otherwise `hard` is dropped
(decision rule set by Caleb, 2026-11-03: at most one calibration iteration).
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fmritest.baselines.fc import fc_components  # noqa: E402
from fmritest.config import apply_overrides, load_config  # noqa: E402
from fmritest.data.generator import generate, zscore_subjects  # noqa: E402
from fmritest.data.splits import make_splits  # noqa: E402
from fmritest.eval.metrics import network_recovery  # noqa: E402

warnings.filterwarnings("ignore")
TARGET, TOL, SEEDS = 0.75, 0.03, [100, 101, 102, 103]
GRID = {"data.overlap_prob": [0.8, 1.0], "data.noise_sd_range": [[1.5, 2.2], [1.8, 2.6], [2.2, 3.2]],
        "data.loading_sd": [0.4, 0.5]}

base = load_config(ROOT / "configs" / "default.yaml")
rows = []
for vals in itertools.product(*GRID.values()):
    ov = dict(zip(GRID, vals))
    cfg = apply_overrides(base, ov)
    sc = []
    for s in SEEDS:
        h = s % 4
        ds = generate(cfg["data"], s, heldout_site=h)
        sp = make_splits(ds.site, h, tuple(cfg["split"]["frac"]), s, group=ds.group, strata=ds.diagnosis)
        Xtr = zscore_subjects(ds.X[sp["train"]])
        sc.append(network_recovery(fc_components(Xtr, 5, "eig_varimax"), ds.membership)["score"])
    rows.append({"overrides": ov, "fc_eig_varimax": float(np.mean(sc)), "per_seed": sc})
    print(json.dumps(rows[-1]), flush=True)
best = min(rows, key=lambda r: abs(r["fc_eig_varimax"] - TARGET))
best["accepted"] = abs(best["fc_eig_varimax"] - TARGET) <= TOL
out = ROOT / "results" / "calibration"
out.mkdir(parents=True, exist_ok=True)
(out / "hard.json").write_text(json.dumps({"target": TARGET, "tol": TOL, "seeds": SEEDS, "grid": rows,
                                           "chosen": best}, indent=1))
print("CHOSEN", json.dumps(best))
