"""Power calibration for the diagnosis repair target, on TRAINING/VALIDATION data only.

For each candidate diagnosis effect size and each calibration seed (disjoint from the reported
sweep seeds), generate data with NO balanced test subjects, pretrain the backbone (full config),
fine-tune vanilla and reweighted diagnosis heads on the train split, and evaluate on `val_balanced`
(extra 50/50-label validation subjects from the training sites).

Targets (decided by Caleb, 2026-11-03): reweighted-minus-vanilla balanced accuracy on the
link-broken set >= 0.25, and per-episode SE of balanced accuracy <= 0.035. The SE target fixes the
test-set size analytically: SE(bacc) = sqrt(p(1-p)/n) <= 0.035 at the worst case p = 0.5 needs
n >= 205, i.e. 70 balanced test subjects per training site (n = 210).

Usage: python scripts/calibrate_power.py --effects -0.5 -0.65 -0.8 --seeds 100 101 --jobs 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fmritest.config import apply_overrides, cap_threads, load_config  # noqa: E402

OUT = ROOT / "outputs" / "calibration"
N_VAL_BAL = 70


def worker(effect: float, seed: int, extra: dict | None = None, tag: str = ""):
    base = load_config(ROOT / "configs" / "default.yaml")
    cfg = apply_overrides(base, {"data.diagnosis.effect": effect, "data.diagnosis.n_balanced_per_site": 0,
                                 **(extra or {})})
    cfg["data"]["diagnosis"]["n_val_balanced_per_site"] = N_VAL_BAL
    cap_threads(cfg["compute"]["n_threads"])
    import numpy as np

    from fmritest.data.generator import generate, zscore_subjects
    from fmritest.data.splits import make_splits
    from fmritest.finetune import evaluate, finetune
    from fmritest.models.train import make_eval_set, train_model

    h = seed % cfg["data"]["n_sites"]
    ds = generate(cfg["data"], seed, heldout_site=h)
    assert not np.any(ds.group == 1), "calibration must not generate test_balanced subjects"
    X = zscore_subjects(ds.X)
    sp = make_splits(ds.site, h, tuple(cfg["split"]["frac"]), seed, group=ds.group, strata=ds.diagnosis)
    # only train / val / val_balanced are used below
    val_set = make_eval_set(X[sp["val"]], cfg["model"]["window"], cfg["eval"]["masks_per_window"],
                            cfg["model"]["mask_ratio"], seed)
    bb, _ = train_model(X[sp["train"]], val_set, cfg, seed, log=lambda s: None)
    fcfg = load_config(ROOT / "configs" / "finetune.yaml")["finetune"]
    sp_val = {"val_balanced": sp["val_balanced"]}
    res = {"effect": effect, "seed": seed, "extra_overrides": extra or {}}
    for v in ("vanilla", "reweight"):
        m, _ = finetune(bb, X, ds.diagnosis, ds.site, sp["train"], fcfg, seed, cfg["compute"]["n_threads"],
                        reweight=(v == "reweight"))
        res[v] = evaluate(m, X, ds.diagnosis, sp_val)["val_balanced"]
    res["headroom"] = res["reweight"]["bacc"] - res["vanilla"]["bacc"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}effect{effect}_seed{seed}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effects", type=float, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--worker", nargs=2)
    ap.add_argument("--extra", default="{}", help="JSON dict of extra dotted-key overrides")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.worker:
        return worker(float(a.worker[0]), int(a.worker[1]), json.loads(a.extra), a.tag)
    jobs = [(e, s) for e in a.effects for s in a.seeds]
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda j: subprocess.call([sys.executable, __file__, "--effects", "0", "--worker",
                                               str(j[0]), str(j[1]), "--extra", a.extra, "--tag", a.tag],
                                              cwd=ROOT), jobs))
    import numpy as np
    rows = ["| effect | seed | vanilla bacc | reweight bacc | headroom | SE (n=%d, worst-case p=0.5) |" % (3 * N_VAL_BAL),
            "|---|---|---|---|---|---|"]
    for e, s in jobs:
        f = OUT / f"{a.tag}effect{e}_seed{s}.json"
        if f.exists():
            r = json.loads(f.read_text())
            rows.append(f"| {e} | {s} | {r['vanilla']['bacc']:.3f} | {r['reweight']['bacc']:.3f} | "
                        f"{r['headroom']:.3f} | {np.sqrt(0.25 / r['vanilla']['n']):.3f} |")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
