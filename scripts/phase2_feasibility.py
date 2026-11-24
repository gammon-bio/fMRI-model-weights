"""Pre-pilot feasibility checks for Phase 2 (requested by the Fable plan review, 2026-11-04).

Uses the frozen-v2 Phase 1 checkpoints (seeds 0-4); these seeds are NOT Phase 2 episode seeds.
1. Oracle-LEACE: on the vanilla fine-tuned diagnosis model, fit (a) a probe on val_balanced labels and
   (b) LEACE-site + probe on val_balanced, then score on sealed test_balanced. If this already reaches
   the reweighted fine-tune, an oracle arm cannot show an agent advantage.
2. Frequency readability: does a simple spectral readout of embed.weight recover the 4 site carrier
   frequencies better than 4 frequencies drawn uniformly in [0, Nyquist]?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fmritest.config import apply_overrides, cap_threads, load_config  # noqa: E402

cfg0 = load_config()
cap_threads(cfg0["compute"]["n_threads"])
from fmritest.data.generator import generate, zscore_subjects  # noqa: E402
from fmritest.data.splits import make_splits  # noqa: E402
from fmritest.eval.diagnosis import diag_eval  # noqa: E402
from fmritest.finetune import DiagnosisModel, subject_embeddings  # noqa: E402
from fmritest.models.transformer import build_model  # noqa: E402


def matched_mae(est, true):
    C = np.abs(np.subtract.outer(est, true))
    r, c = linear_sum_assignment(C)
    return float(C[r, c].mean())


out = {"oracle_leace": [], "freq": []}
rng = np.random.default_rng(12345)
for seed in cfg0["seeds"]:
    for cond in ("confound_on", "confound_off"):
        cfg = apply_overrides(cfg0, cfg0["conditions"][cond])
        h = seed % 4
        ds = generate(cfg["data"], seed, heldout_site=h)
        # --- frequency readout from the pretrained backbone's embed.weight (d x W)
        bb = build_model(cfg["model"], 30)
        bb.load_state_dict(torch.load(ROOT / f"outputs/runs/{cond}/seed{seed}/model.pt"))
        Wemb = bb.embed.weight.detach().numpy()             # (d, W): each row a temporal filter
        W = Wemb.shape[1]
        P = (np.abs(np.fft.rfft(Wemb - Wemb.mean(1, keepdims=True), n=8 * W, axis=1)) ** 2).sum(0)
        f = np.fft.rfftfreq(8 * W, d=cfg["data"]["tr"])
        # 4 largest spectral peaks above 0.02 Hz, separated by >= 0.02 Hz
        order = np.argsort(P)[::-1]
        peaks = []
        for i in order:
            if f[i] < 0.02:
                continue
            if all(abs(f[i] - p) >= 0.02 for p in peaks):
                peaks.append(f[i])
            if len(peaks) == 4:
                break
        nyq = 0.5 / cfg["data"]["tr"]
        null = [matched_mae(rng.uniform(0, nyq, 4), ds.site_freqs) for _ in range(2000)] if cond == "confound_on" else []
        rec = {"seed": seed, "cond": cond, "peaks_hz": [round(p, 4) for p in peaks]}
        if cond == "confound_on":
            mae = matched_mae(np.array(peaks), ds.site_freqs)
            rec.update(true_hz=[round(x, 4) for x in sorted(ds.site_freqs)], mae=mae,
                       null_mean=float(np.mean(null)), p=float((1 + np.sum(np.array(null) <= mae)) / 2001))
        out["freq"].append(rec)
        if cond != "confound_on":
            continue
        # --- oracle-LEACE on the vanilla fine-tuned diagnosis model
        X = zscore_subjects(ds.X)
        sp = make_splits(ds.site, h, tuple(cfg["split"]["frac"]), seed, group=ds.group, strata=ds.diagnosis)
        m = DiagnosisModel.from_backbone(bb)
        m.load_state_dict(torch.load(ROOT / f"outputs/finetune/{cond}/seed{seed}/vanilla.pt"))
        m.eval()
        E = {k: subject_embeddings(m, X[v]) for k, v in sp.items()}
        y, site = ds.diagnosis, ds.site
        # treat val_balanced as the probe-training split
        sp_o = dict(sp, train=sp["val_balanced"])
        E_o = dict(E, train=E["val_balanced"])
        r_probe = diag_eval(E_o, y, site, sp_o, 0.1, seed)
        r_leace = diag_eval(E_o, y, site, sp_o, 0.1, seed, erase_site=True)
        ft = json.loads((ROOT / f"results/finetune/{cond}_seed{seed}.json").read_text())
        out["oracle_leace"].append({
            "seed": seed,
            "vanilla": ft["vanilla"]["test_balanced"]["bacc"],
            "reweight_ft": ft["reweight"]["test_balanced"]["bacc"],
            "posthoc_leace_train": ft["vanilla"]["posthoc"]["probe+leace_site"]["test_balanced"]["bacc"],
            "oracle_probe_valbal": r_probe["test_balanced"]["bacc"],
            "oracle_leace_valbal": r_leace["test_balanced"]["bacc"]})
        print(json.dumps(out["oracle_leace"][-1]), json.dumps(rec), flush=True)
dst = ROOT / "results" / "phase2_feasibility.json"
dst.write_text(json.dumps(out, indent=1))
print("wrote", dst)
