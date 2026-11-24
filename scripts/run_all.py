"""Run every (condition, seed) in the config. Each run is a separate process with capped threads.

Usage: python scripts/run_all.py [--config configs/default.yaml] [--jobs 1] [--only confound_on:0]
Per-run artifacts (checkpoints, intervention logs) -> outputs/runs/; small JSON summaries -> results/runs/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fmritest.config import load_config  # noqa: E402


def one(config: str, cond: str, seed: int) -> int:
    log = ROOT / "outputs" / "logs" / f"{cond}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "run_seed.py"), "--config", config,
           "--condition", cond, "--seed", str(seed)]
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    print(f"{cond} seed {seed}: exit {rc} (log {log.relative_to(ROOT)})", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None, help="cond:seed items")
    a = ap.parse_args()
    cfg = load_config(a.config)
    jobs = [(c, s) for c in cfg["conditions"] for s in cfg["seeds"]]
    if a.only:
        jobs = [(c, int(s)) for c, s in (o.split(":") for o in a.only)]
    if a.jobs * cfg["compute"]["n_threads"] > (__import__("os").cpu_count() or 1):
        sys.exit(f"refusing: jobs*n_threads exceeds the core count")
    with ThreadPoolExecutor(a.jobs) as ex:
        rcs = list(ex.map(lambda j: one(a.config, *j), jobs))
    if any(rcs):
        sys.exit(f"{sum(r != 0 for r in rcs)} run(s) failed; see outputs/logs/")


if __name__ == "__main__":
    main()
