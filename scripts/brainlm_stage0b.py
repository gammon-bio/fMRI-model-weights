"""BrainLM Stage 0b (plan: docs/brainlm_stage0b_plan.md, fixed in 6bfab91). Synthetic plumbing only.

  .venv/bin/python scripts/brainlm_stage0b.py tune [--seeds 300 301]     # C5: fine-tune + val_balanced only
  .venv/bin/python scripts/brainlm_stage0b.py gate --effect -0.4         # gate seeds 210-212 with controls a-e
  .venv/bin/python scripts/brainlm_stage0b.py summarize                  # gates from per_seed.json

Changes vs Stage 0 (scripts/brainlm_stage0.py stays as it was, so Stage 0 is reproducible):
C1 per-seed carrier bins (data.confound.freq_choices), C2 group-balanced reference, C3 fixed 10 epochs
with final weights, C4 prevalence 0.8/0.2/0.8 and 150 subjects/site, C5 effect from tuning seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG0B = yaml.safe_load((ROOT / "configs" / "brainlm_stage0b.yaml").read_text())
N_THREADS = int(CFG0B["compute"]["n_threads"])
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = str(N_THREADS)

import numpy as np
import torch

sys.path.insert(0, str(ROOT))
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from fmritest.brainlm import carriers as C
from fmritest.brainlm import model as M
from fmritest.brainlm import pack as P
from fmritest.config import apply_overrides, load_config
from fmritest.data.generator import generate
from fmritest.data.splits import make_splits
from fmritest.eval.diagnosis import balance_weights

torch.set_num_threads(N_THREADS)
OUT = ROOT / "results" / "brainlm_stage0b"
AMP = bool(CFG0B["compute"]["bf16_autocast"])
EVAL = ("test_balanced", "test", "test_site", "val_balanced")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auc(y, s) -> float:
    return float(roc_auc_score(y, s))


def make_cfg(effect: float) -> dict:
    cfg = load_config(ROOT / CFG0B["base_config"])
    over = dict(CFG0B["data_overrides"])
    choices = over.pop("data.confound.freq_choices")  # new opt-in key: apply_overrides only edits existing keys
    over["data.diagnosis.effect"] = float(effect)
    cfg = apply_overrides(cfg, over)
    cfg["data"]["confound"]["freq_choices"] = list(choices)
    return cfg


def prepare(seed: int, effect: float, enc) -> dict:
    """Generate, scale, cache, fine-tune vanilla + group-balanced reference, evaluate heads."""
    T = {}
    cfg = make_cfg(effect)
    heldout = seed % cfg["data"]["n_sites"]
    t = time.time()
    ds = generate(cfg["data"], seed, heldout_site=heldout)
    sp = make_splits(ds.site, heldout, tuple(cfg["split"]["frac"]), seed, group=ds.group, strata=ds.diagnosis)
    T["generate"] = time.time() - t
    y, site = ds.diagnosis, ds.site
    train_sites = [c for c in range(cfg["data"]["n_sites"]) if c != heldout]
    low_site = train_sites[int(np.argmin(cfg["data"]["diagnosis"]["train_prevalence"]))]
    freqs = ds.site_freqs
    tr, va = sp["train"], sp["val"]
    cells = {f"site{c}_y{v}": int(np.sum((site[tr] == c) & (y[tr] == v))) for c in train_sites for v in (0, 1)}
    log(f"seed {seed} effect {effect}: heldout {heldout}, low-prev site {low_site}, freqs {np.round(freqs, 3).tolist()}, "
        + ", ".join(f"{k}={len(v)}" for k, v in sp.items()) + f"; train cells {cells}")
    t = time.time()
    med, iqr = P.fit_robust_scaler(ds.X[tr])
    Xs = P.robust_scale(ds.X, med, iqr).astype(np.float32)
    del ds.X, ds.signal, ds.noise, ds.confound
    T["scale"] = time.time() - t
    idx = M.token_index(P.data_patch_mask(cfg["data"]["n_regions"], cfg["data"]["n_timepoints"]))
    t = time.time()
    H = M.encode_recordings(enc, Xs, amp=AMP)
    T["cache"] = time.time() - t
    log(f"  cache {tuple(H.shape)} in {T['cache']:.0f}s")
    ft = dict(CFG0B["finetune"])
    res = {"seed": seed, "effect": effect, "heldout_site": heldout, "low_prevalence_site": int(low_site),
           "site_freqs": freqs.tolist(), "train_cells": cells, "finetune": ft, "reference": CFG0B["reference"],
           "n": {k: len(v) for k, v in sp.items()}, "models": {}}
    models = {}
    for variant in ("vanilla", "reweighted"):
        ref = variant == "reweighted"
        groups = site[tr] * 2 + y[tr] if ref else None          # C2: group-balanced sampling
        w_va = balance_weights(site[va], y[va]) if ref else np.ones(len(va))   # reporting only
        log(f"  fine-tune {variant} ({'group-balanced sampling' if ref else 'plain'}, {ft['epochs']} epochs, final weights)")
        t = time.time()
        m, info = M.train_head(enc, H[tr], y[tr], np.ones(len(tr)), H[va], y[va], w_va, idx, ft,
                               seed + (50 if ref else 0), amp=AMP, log=log, group_ids=groups)
        T[f"finetune_{variant}"] = time.time() - t
        r = {"train_info": info}
        t = time.time()
        for k in EVAL:
            s = M.predict(m, H[sp[k]], amp=AMP)
            r[k] = {"auc": auc(y[sp[k]], s), "bacc": float(balanced_accuracy_score(y[sp[k]], (s > 0).astype(int))),
                    "n": len(sp[k])}
        T[f"eval_head_{variant}"] = time.time() - t
        res["models"][variant] = r
        models[variant] = m
        log(f"  {variant}: AUC test_bal {r['test_balanced']['auc']:.3f} test {r['test']['auc']:.3f} "
            f"test_site {r['test_site']['auc']:.3f} val_bal {r['val_balanced']['auc']:.3f}")
    del H
    mv, mr = res["models"]["vanilla"], res["models"]["reweighted"]
    res["headroom_val_balanced"] = mr["val_balanced"]["auc"] - mv["val_balanced"]["auc"]
    res["headroom_test_balanced"] = mr["test_balanced"]["auc"] - mv["test_balanced"]["auc"]
    T["prepare"] = T["generate"] + T["scale"] + T["cache"] + T["finetune_vanilla"] + T["finetune_reweighted"]
    res["timing_s"] = T
    return {"res": res, "Xs": Xs, "sp": sp, "y": y, "models": models, "freqs": freqs, "low_site": low_site}


def cmd_tune(args, enc):
    tc = CFG0B["tuning"]
    seeds = args.seeds or tc["seeds"]
    rows, t_all = [], time.time()
    path = OUT / "tuning.json"
    for i, seed in enumerate(seeds):
        for effect in tc["effects"]:
            t = time.time()
            p = prepare(seed, effect, enc)
            r = p["res"]
            r["timing_s"]["wall"] = time.time() - t
            rows.append({k: r[k] for k in ("seed", "effect", "headroom_val_balanced", "site_freqs", "train_cells",
                                            "heldout_site", "timing_s")}
                        | {"val_balanced": {v: r["models"][v]["val_balanced"] for v in r["models"]},
                           "val_loss_final": {v: r["models"][v]["train_info"]["best_val_loss"] for v in r["models"]}})
            log(f"  TUNE seed {seed} effect {effect}: val_balanced headroom {r['headroom_val_balanced']:+.3f} "
                f"({r['timing_s']['wall']:.0f}s)")
            path.write_text(json.dumps({"rows": rows}, indent=1))
        if i == 0:
            per_seed = time.time() - t_all
            log(f"  EXTRAPOLATION after first tuning seed: {per_seed:.0f}s per tuning seed")
    hr = {e: float(np.mean([r["headroom_val_balanced"] for r in rows if r["effect"] == e])) for e in tc["effects"]}
    a, b = tc["effects"]
    if abs(hr[a] - hr[b]) <= tc["tie_tol"]:
        chosen, why = tc["tie_winner"], f"tie within {tc['tie_tol']}: {tc['tie_winner']}"
    else:
        chosen = max(hr, key=hr.get)
        why = "larger mean val_balanced headroom"
    sel = {"rule": "choose the effect with the larger mean val_balanced headroom over the tuning seeds; "
                   "a tie within 0.01 goes to -0.4 (plan C5)",
           "mean_val_balanced_headroom": {str(k): v for k, v in hr.items()}, "chosen_effect": chosen, "reason": why,
           "wall_s": time.time() - t_all}
    path.write_text(json.dumps({"rows": rows, "selection": sel}, indent=1))
    log(json.dumps(sel))


def cmd_gate(args, enc):
    seeds = args.seeds or CFG0B["gate_seeds"]
    per_seed, path = [], OUT / "per_seed.json"
    kmin = int(CFG0B["controls"]["band_kmin"])
    B = C.band_basis(kmin)
    cand = list(CFG0B["data_overrides"]["data.confound.freq_choices"])
    for seed in seeds:
        t_seed = time.time()
        p = prepare(seed, args.effect, enc)
        res, Xs, sp, y, van = p["res"], p["Xs"], p["sp"], p["y"], p["models"]["vanilla"]
        T = res["timing_s"]
        freqs = p["freqs"]
        Qc = C.carrier_basis(freqs)
        unused = [f for f in cand if not np.any(np.isclose(f, freqs))]
        assert len(unused) == 2
        projs = {"a_all_carriers": Qc, "b_low_site_carrier": C.carrier_basis([freqs[p["low_site"]]]),
                 "d_unused_bins": C.carrier_basis(unused), "e_lowpass_k2plus": B}
        rng = np.random.default_rng([seed, 99])
        for i in range(int(CFG0B["controls"]["n_random"])):
            projs[f"c_random_inband_{i:02d}"] = C.random_in_span(rng, B, Qc.shape[1])
        W0 = M.kernel(enc)
        sub = np.concatenate([sp["test_balanced"], sp["test"]])
        ntb = len(sp["test_balanced"])

        def scores(W, Xs=Xs, sub=sub, van=van, y=y, sp=sp, ntb=ntb):
            with M.edited_kernel(enc, W):
                He = M.encode_recordings(enc, Xs[sub], amp=AMP, out_dtype=torch.float32)
            s = M.predict(van, He, amp=AMP)
            return auc(y[sp["test_balanced"]], s[:ntb]), auc(y[sp["test"]], s[ntb:])

        t = time.time()
        tb0, te0 = scores(W0)
        T["edit_forward_unedited"] = time.time() - t
        pc = {"unedited": {"test_balanced_auc": tb0, "test_auc": te0}, "unused_bins_hz": unused, "edits": {}}
        log(f"  unedited full forward: test_bal {tb0:.3f} (cache {res['models']['vanilla']['test_balanced']['auc']:.3f}), "
            f"test {te0:.3f}, {T['edit_forward_unedited']:.0f}s")
        T["edit_forward"] = {}
        W0np = W0.numpy().astype(np.float64)
        for name, Q in projs.items():
            t = time.time()
            W1 = C.project_out(W0, Q)
            tb, te = scores(W1)
            T["edit_forward"][name] = time.time() - t
            rho = C.carrier_share(W1.numpy().astype(np.float64) - W0np, Qc)
            pc["edits"][name] = {"dims": int(Q.shape[1]), "delta_test_balanced_auc": tb - tb0, "delta_test_auc": te - te0,
                                 "test_balanced_auc": tb, "test_auc": te, "rho": rho,
                                 "subspace_carrier_overlap": C.subspace_overlap(Qc, Q)}
            log(f"    {name}: d test_bal {tb - tb0:+.3f}, d test {te - te0:+.3f}, rho {rho:.2f} ({T['edit_forward'][name]:.0f}s)")
        rnd = [v for k, v in pc["edits"].items() if k.startswith("c_")]
        d_c = np.array([v["delta_test_balanced_auc"] for v in rnd])
        rho_c = np.array([v["rho"] for v in rnd])
        da = pc["edits"]["a_all_carriers"]["delta_test_balanced_auc"]
        pc["c_summary"] = {"delta_mean": float(d_c.mean()), "delta_sd": float(d_c.std(ddof=1)), "delta_max": float(d_c.max()),
                           "delta_min": float(d_c.min()), "a_percentile": float(100 * np.mean(d_c < da)),
                           "rho_mean": float(rho_c.mean()), "rho_sd": float(rho_c.std(ddof=1)),
                           "rho_min": float(rho_c.min()), "rho_max": float(rho_c.max()),
                           "rho_quantiles_5_50_95": np.quantile(rho_c, [0.05, 0.5, 0.95]).tolist()}
        res["positive_control"] = pc
        T["controls"] = T["edit_forward_unedited"] + sum(T["edit_forward"].values())
        T["prepare_plus_controls"] = T["prepare"] + T["controls"]
        T["seed_wall"] = time.time() - t_seed
        per_seed.append(res)
        path.write_text(json.dumps(per_seed, indent=1))
        log(f"seed {seed} done in {T['seed_wall']:.0f}s (prepare {T['prepare']:.0f}, controls {T['controls']:.0f})")
    cmd_summarize(args)


def gates(R: list[dict]) -> dict:
    v_tb = np.array([r["models"]["vanilla"]["test_balanced"]["auc"] for r in R])
    v_te = np.array([r["models"]["vanilla"]["test"]["auc"] for r in R])
    r_tb = np.array([r["models"]["reweighted"]["test_balanced"]["auc"] for r in R])
    r_te = np.array([r["models"]["reweighted"]["test"]["auc"] for r in R])
    da = np.array([r["positive_control"]["edits"]["a_all_carriers"]["delta_test_balanced_auc"] for r in R])
    de = np.array([r["positive_control"]["edits"]["e_lowpass_k2plus"]["delta_test_balanced_auc"] for r in R])
    cm = np.array([r["positive_control"]["c_summary"]["delta_mean"] for r in R])
    cx = np.array([r["positive_control"]["c_summary"]["delta_max"] for r in R])
    need = min(2, len(R))
    g_short = {"mean_headroom": float(np.mean(r_tb - v_tb)), "headroom": (r_tb - v_tb).tolist(),
               "mean_vanilla_gap": float(np.mean(v_te - v_tb)), "vanilla_gap": (v_te - v_tb).tolist(),
               "reweighted_test_auc": r_te.tolist()}
    g_short["pass"] = bool(g_short["mean_headroom"] >= 0.15 and g_short["mean_vanilla_gap"] >= 0.10 and np.all(r_te >= 0.70))
    g_ctrl = {"mean_a_minus_c_mean": float(np.mean(da - cm)), "a_minus_c_mean": (da - cm).tolist(),
              "a_gt_c_max": (da > cx).tolist(), "a_gt_0": (da > 0).tolist(), "delta_a": da.tolist(),
              "c_mean": cm.tolist(), "c_max": cx.tolist()}
    g_ctrl["pass"] = bool(g_ctrl["mean_a_minus_c_mean"] >= 0.05 and np.sum(da > cx) >= need and np.sum(da > 0) >= need)
    g_spec = {"mean_a_minus_e": float(np.mean(da - de)), "a_minus_e": (da - de).tolist(), "delta_e": de.tolist()}
    g_spec["pass"] = bool(g_spec["mean_a_minus_e"] >= 0.03 and np.sum(da - de > 0) >= need)
    secs = [r["timing_s"]["prepare_plus_controls"] for r in R]
    return {"G0b_shortcut": g_short, "G0b_control": g_ctrl, "G0b_specific": g_spec,
            "G0b_compute": {"seconds_per_seed_prepare_plus_controls": secs, "mean": float(np.mean(secs)), "pass": True}}


def cmd_summarize(args):
    R = json.loads((OUT / "per_seed.json").read_text())
    tun = json.loads((OUT / "tuning.json").read_text()) if (OUT / "tuning.json").exists() else {}
    summary = {"plan": "docs/brainlm_stage0b_plan.md (6bfab91)", "seeds": [r["seed"] for r in R],
               "effect": R[0]["effect"], "selection": tun.get("selection"), "gates": gates(R),
               "note": "Synthetic plumbing only; not a scientific result."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary["gates"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "gate", "summarize"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--effect", type=float, default=None, help="gate: the effect chosen in tuning.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "summarize":
        return cmd_summarize(args)
    if args.cmd == "gate":
        sel = json.loads((OUT / "tuning.json").read_text())["selection"]
        if args.effect is None:
            args.effect = sel["chosen_effect"]
        if args.effect != sel["chosen_effect"]:
            raise SystemExit(f"--effect {args.effect} differs from the tuning selection {sel['chosen_effect']}")
    t = time.time()
    enc = M.load_encoder(CFG0B["weights_dir"])
    log(f"encoder loaded + sha256 verified in {time.time() - t:.1f}s; threads {torch.get_num_threads()}; bf16 {AMP}")
    (cmd_tune if args.cmd == "tune" else cmd_gate)(args, enc)


if __name__ == "__main__":
    main()
