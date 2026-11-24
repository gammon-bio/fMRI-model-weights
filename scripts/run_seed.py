"""Run a single (condition, seed). Called by run_all.py; usable on its own."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fmritest.config import apply_overrides, cap_threads, load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
ap.add_argument("--condition", required=True)
ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()
cfg = load_config(a.config)
cap_threads(cfg["compute"]["n_threads"])
from fmritest.pipeline import run_seed  # noqa: E402

out = ROOT / "outputs" / "runs" / a.condition / f"seed{a.seed}"
cfg_c = apply_overrides(cfg, cfg["conditions"][a.condition])
res = run_seed(cfg_c, a.seed, out, log=lambda s: print(s, flush=True))
res["condition"] = a.condition
dst = ROOT / "results" / "runs"
dst.mkdir(parents=True, exist_ok=True)
(dst / f"{a.condition}_seed{a.seed}.json").write_text(json.dumps(res, indent=1))
shutil.copy(out / "curves.json", dst / f"{a.condition}_seed{a.seed}_curves.json")
