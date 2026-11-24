"""BrainLM Stage 0: synthetic 424-parcel plumbing with the real vitmae_111M weights.

Per seed: generate -> robust-scale + pack -> cache layer-9 output (input to layer 10) -> fine-tune
layers 10-11 + head (vanilla and site x label reweighted) -> AUC on test_balanced / test / test_site
(+ val_balanced, used only for setting selection) -> site probe on the head input -> positive control
on the vanilla model: patch-kernel temporal projections, each with a full forward of test_balanced.

Not a scientific result: the data are synthetic. Weights stay outside the repo (CC BY-NC-ND 4.0);
fine-tuned weights and caches are never written into the repo.

  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/brainlm_stage0.py --seeds 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "2"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fmritest.brainlm import carriers as C
from fmritest.brainlm import model as M
from fmritest.brainlm import pack as P
from fmritest.config import apply_overrides, load_config
from fmritest.data.generator import generate
from fmritest.data.splits import make_splits
from fmritest.eval.diagnosis import balance_weights

N_THREADS = 2
torch.set_num_threads(N_THREADS)

WEIGHTS = Path("/home/user/brainlm_scout/weights/vitmae_111M")
DATA_OVERRIDES = {
    "data.n_regions": 424,
    "data.tr": 0.72,
    "data.n_timepoints": 200,
    "data.confound.freq_range": [0.2, 0.6],
}
FT = {"epochs": 8, "batch_size": 16, "lr_head": 1e-3, "lr_layers": 5e-5, "weight_decay": 0.01, "patience": 2}
OFF_FREQS = [0.27, 0.40, 0.53, 0.13]
EVAL = ("test_balanced", "test", "test_site", "val_balanced")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auc(y, s):
    return float(roc_auc_score(y, s))


def site_probe(F, site, sp):
    """Multinomial logistic site probe: fit on train, accuracy on test (training-site subjects)."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    clf.fit(F[sp["train"]], site[sp["train"]])
    return {"train_acc": float(clf.score(F[sp["train"]], site[sp["train"]])),
            "test_acc": float(clf.score(F[sp["test"]], site[sp["test"]])),
            "test_bacc": float(balanced_accuracy_score(site[sp["test"]], clf.predict(F[sp["test"]]))),
            "chance": 1.0 / len(np.unique(site[sp["train"]]))}


def run_seed(seed: int, enc, args, data_over: dict, ft: dict) -> dict:
    T = {}
    amp = not args.fp32
    cfg = apply_overrides(load_config(ROOT / "configs" / "default.yaml"), data_over)
    heldout = seed % cfg["data"]["n_sites"]

    t = time.time()
    ds = generate(cfg["data"], seed, heldout_site=heldout)
    sp = make_splits(ds.site, heldout, tuple(cfg["split"]["frac"]), seed, group=ds.group, strata=ds.diagnosis)
    T["generate"] = time.time() - t
    y, site = ds.diagnosis, ds.site
    train_sites = [c for c in range(cfg["data"]["n_sites"]) if c != heldout]
    low_site = train_sites[int(np.argmin(cfg["data"]["diagnosis"]["train_prevalence"]))]
    freqs = ds.site_freqs
    log(f"seed {seed}: heldout {heldout}, low-prevalence site {low_site}, site freqs {np.round(freqs, 3).tolist()}, "
        f"n " + ", ".join(f"{k}={len(v)}" for k, v in sp.items()))

    t = time.time()
    med, iqr = P.fit_robust_scaler(ds.X[sp["train"]])
    Xs = P.robust_scale(ds.X, med, iqr).astype(np.float32)
    del ds.X, ds.signal, ds.noise, ds.confound
    t_pack = time.time()
    P.pack(Xs[:32])
    T["scale"] = t_pack - t
    T["pack_per_32"] = time.time() - t_pack
    mask = P.data_patch_mask(cfg["data"]["n_regions"], cfg["data"]["n_timepoints"])
    idx = M.token_index(mask)

    t = time.time()
    H = M.encode_recordings(enc, Xs, amp=amp)
    T["cache"] = time.time() - t
    T["cache_per_sample"] = T["cache"] / len(Xs)
    log(f"  cache {tuple(H.shape)} {H.dtype} in {T['cache']:.0f}s")

    res = {"seed": seed, "heldout_site": heldout, "low_prevalence_site": int(low_site),
           "site_freqs": freqs.tolist(), "data_overrides": data_over, "finetune": ft, "bf16_autocast": amp,
           "n": {k: len(v) for k, v in sp.items()}, "models": {}}
    tr, va = sp["train"], sp["val"]
    models = {}
    for variant in ("vanilla", "reweighted"):
        rw = variant == "reweighted"
        w_tr = balance_weights(site[tr], y[tr]) if rw else np.ones(len(tr))
        w_va = balance_weights(site[va], y[va]) if rw else np.ones(len(va))
        log(f"  fine-tune {variant}")
        t = time.time()
        m, info = M.train_head(enc, H[tr], y[tr], w_tr, H[va], y[va], w_va, idx, ft, seed + (0 if not rw else 50),
                               amp=amp, log=log)
        T[f"finetune_{variant}"] = time.time() - t
        r = {"train_info": info}
        for k in EVAL:
            t = time.time()
            s = M.predict(m, H[sp[k]], amp=amp)
            r[k] = {"auc": auc(y[sp[k]], s), "bacc": float(balanced_accuracy_score(y[sp[k]], (s > 0).astype(int))),
                    "n": len(sp[k])}
            T[f"eval_head_{variant}_{k}"] = time.time() - t
        t = time.time()
        Fh = M.predict(m, H, feats=True, amp=amp)
        r["site_probe_head_input"] = site_probe(Fh, site, sp)
        T[f"site_probe_{variant}"] = time.time() - t
        res["models"][variant] = r
        models[variant] = m
        log(f"  {variant}: AUC test_bal {r['test_balanced']['auc']:.3f} test {r['test']['auc']:.3f} "
            f"test_site {r['test_site']['auc']:.3f} val_bal {r['val_balanced']['auc']:.3f}; "
            f"site probe {r['site_probe_head_input']['test_acc']:.2f}")
    F9 = H[:, idx].float().mean(1).numpy()
    res["site_probe_layer9_mean"] = site_probe(F9, site, sp)
    del H
    res["headroom_test_balanced"] = res["models"]["reweighted"]["test_balanced"]["auc"] - res["models"]["vanilla"]["test_balanced"]["auc"]
    res["headroom_val_balanced"] = res["models"]["reweighted"]["val_balanced"]["auc"] - res["models"]["vanilla"]["val_balanced"]["auc"]

    T["prepare"] = T["generate"] + T["scale"] + T["cache"] + T["finetune_vanilla"] + T["finetune_reweighted"]
    if args.no_control:
        res["timing_s"] = T
        return res
    # ---- positive control on vanilla: edited patch kernel, full forward of test_balanced
    van = models["vanilla"]
    W0 = M.kernel(enc)
    Qc = C.carrier_basis(freqs)
    projs = {"a_all_carriers": Qc, "b_low_site_carrier": C.carrier_basis([freqs[low_site]])}
    for f in OFF_FREQS:
        projs[f"d_off_{f:.2f}"] = C.carrier_basis([f])
    projs["d_off_all4"] = C.carrier_basis(OFF_FREQS)
    rng = np.random.default_rng([seed, 99])
    for i in range(args.n_random):
        projs[f"c_random_{i:02d}"] = C.random_basis(rng, Qc.shape[1])
    tb = sp["test_balanced"]

    def edited_logits(W, sub):
        with M.edited_kernel(enc, W):
            He = M.encode_recordings(enc, Xs[sub], amp=amp, out_dtype=torch.float32)
        return M.predict(van, He, amp=amp)

    t = time.time()
    s0 = edited_logits(W0, tb)
    T["edit_forward_unedited"] = time.time() - t
    auc0 = auc(y[tb], s0)
    pc = {"unedited_full_forward_auc": auc0,
          "unedited_cache_auc": res["models"]["vanilla"]["test_balanced"]["auc"], "edits": {}}
    log(f"  unedited full-forward test_bal AUC {auc0:.3f} (cache path {pc['unedited_cache_auc']:.3f}), {T['edit_forward_unedited']:.0f}s")
    T["edit_forward"] = {}
    for name, Q in projs.items():
        t = time.time()
        s = edited_logits(C.project_out(W0, Q), tb)
        a = auc(y[tb], s)
        T["edit_forward"][name] = time.time() - t
        pc["edits"][name] = {"dims": int(Q.shape[1]), "auc": a, "delta_auc": a - auc0,
                             "carrier_overlap": C.subspace_overlap(Qc, Q),
                             "mean_abs_logit_shift": float(np.mean(np.abs(s - s0)))}
        log(f"    {name}: AUC {a:.3f} (delta {a - auc0:+.3f}, overlap {pc['edits'][name]['carrier_overlap']:.2f}) "
            f"{T['edit_forward'][name]:.0f}s")
    # in-distribution test AUC with and without the full carrier projection (does the shortcut go away?)
    te = sp["test"]
    t = time.time()
    pc["test_unedited_auc"] = auc(y[te], edited_logits(W0, te))
    pc["test_a_all_carriers_auc"] = auc(y[te], edited_logits(C.project_out(W0, Qc), te))
    T["edit_forward_test_pair"] = time.time() - t
    rnd = np.array([v["delta_auc"] for k, v in pc["edits"].items() if k.startswith("c_random")])
    da = pc["edits"]["a_all_carriers"]["delta_auc"]
    pc["random_summary"] = {"n": len(rnd), "mean": float(rnd.mean()), "sd": float(rnd.std(ddof=1)),
                            "max": float(rnd.max()), "a_minus_random_mean": float(da - rnd.mean()),
                            "a_percentile": float(100.0 * np.mean(rnd < da)),
                            "a_beats_all": bool(np.all(da > rnd))}
    if args.fp32_check and seed == args.seeds[0] and amp:
        t = time.time()
        with torch.autocast("cpu", enabled=False):
            def fp32_logits(W):
                with M.edited_kernel(enc, W):
                    He = M.encode_recordings(enc, Xs[tb], amp=False, out_dtype=torch.float32)
                return M.predict(van, He, amp=False)
            a32_0 = auc(y[tb], fp32_logits(W0))
            a32_a = auc(y[tb], fp32_logits(C.project_out(W0, Qc)))
        pc["fp32_check"] = {"unedited_auc": a32_0, "a_all_carriers_auc": a32_a, "delta_auc": a32_a - a32_0,
                            "bf16_delta_auc": da}
        T["fp32_check"] = time.time() - t
        log(f"  fp32 check: unedited {a32_0:.3f}, (a) {a32_a:.3f}, delta {a32_a - a32_0:+.3f} vs bf16 {da:+.3f}")
    res["positive_control"] = pc
    T["nulls_and_controls"] = T["edit_forward_unedited"] + sum(T["edit_forward"].values())
    T["per_projection_mean"] = float(np.mean(list(T["edit_forward"].values())))
    res["timing_s"] = T
    return res


def gates(per_seed: list[dict]) -> dict:
    hr = [r["headroom_test_balanced"] for r in per_seed]
    v_te = [r["models"]["vanilla"]["test"]["auc"] for r in per_seed]
    v_tb = [r["models"]["vanilla"]["test_balanced"]["auc"] for r in per_seed]
    hr_val = [r["headroom_val_balanced"] for r in per_seed]
    out = {"G0_shortcut": {"mean_headroom": float(np.mean(hr)), "headroom": hr, "vanilla_test_auc": v_te,
                           "vanilla_test_balanced_auc": v_tb, "headroom_val_balanced": hr_val,
                           "mean_headroom_val_balanced": float(np.mean(hr_val)),
                           "mean_in_dist_minus_balanced": float(np.mean(np.subtract(v_te, v_tb))),
                           "pass": bool(np.mean(hr) >= 0.15 and np.mean(np.subtract(v_te, v_tb)) >= 0.10)}}
    if not all("positive_control" in r for r in per_seed):
        return out
    a_minus = [r["positive_control"]["random_summary"]["a_minus_random_mean"] for r in per_seed]
    beats = [r["positive_control"]["random_summary"]["a_beats_all"] for r in per_seed]
    secs = [r["timing_s"]["prepare"] + r["timing_s"]["nulls_and_controls"] for r in per_seed]
    return {
        "G0_shortcut": {"mean_headroom": float(np.mean(hr)), "headroom": hr, "vanilla_test_auc": v_te,
                        "vanilla_test_balanced_auc": v_tb,
                        "mean_in_dist_minus_balanced": float(np.mean(np.subtract(v_te, v_tb))),
                        "pass": bool(np.mean(hr) >= 0.15 and np.mean(np.subtract(v_te, v_tb)) >= 0.10)},
        "G0_control": {"mean_a_minus_random_mean": float(np.mean(a_minus)), "per_seed": a_minus,
                       "a_beats_all_random": beats,
                       "pass": bool(np.mean(a_minus) >= 0.05 and sum(beats) >= min(2, len(beats)))},
        "G0_compute": {"seconds_per_seed_prepare_plus_controls": secs, "mean": float(np.mean(secs)), "pass": True},
    }


def merge(out: Path, tags: list[str]) -> None:
    per_seed = []
    for tag in tags:
        per_seed += json.loads((out / f"per_seed_{tag}.json").read_text())
    per_seed.sort(key=lambda r: r["seed"])
    (out / "per_seed.json").write_text(json.dumps(per_seed, indent=1))
    r0 = per_seed[0]
    summary = {"seeds": [r["seed"] for r in per_seed], "merged_from": tags, "gates": gates(per_seed),
               "data_overrides": r0["data_overrides"], "finetune": r0["finetune"], "bf16_autocast": r0["bf16_autocast"],
               "note": "Synthetic plumbing only; not a scientific result."}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["gates"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[200, 201, 202])
    ap.add_argument("--out", default=str(ROOT / "results" / "brainlm_stage0"))
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--fp32", action="store_true", help="disable bf16 autocast everywhere")
    ap.add_argument("--fp32-check", action="store_true", help="first seed: re-run unedited and (a) in fp32")
    ap.add_argument("--set", nargs="*", default=[], help="extra data overrides key=value (yaml-parsed)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-control", action="store_true", help="fine-tune + eval only (setting selection)")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="combine existing per_seed_<tag>.json files into per_seed.json + summary.json (no compute)")
    args = ap.parse_args()
    if args.merge:
        merge(Path(args.out), args.merge)
        return
    import yaml
    data_over = dict(DATA_OVERRIDES)
    for kv in args.set:
        k, v = kv.split("=", 1)
        data_over[k] = yaml.safe_load(v)
    ft = dict(FT)
    if args.epochs:
        ft["epochs"] = args.epochs
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    t = time.time()
    enc = M.load_encoder(WEIGHTS)
    t_load = time.time() - t
    log(f"encoder loaded + sha256 verified in {t_load:.1f}s; threads {torch.get_num_threads()}")
    per_seed = []
    path = out / f"per_seed{suffix}.json"
    for seed in args.seeds:
        t = time.time()
        r = run_seed(seed, enc, args, data_over, ft)
        r["timing_s"]["seed_wall"] = time.time() - t
        per_seed.append(r)
        path.write_text(json.dumps(per_seed, indent=1))
        log(f"seed {seed} done in {r['timing_s']['seed_wall']:.0f}s")
    summary = {"seeds": args.seeds, "encoder_load_s": t_load, "gates": gates(per_seed),
               "data_overrides": data_over, "finetune": ft, "bf16_autocast": not args.fp32,
               "note": "Synthetic plumbing only; not a scientific result."}
    (out / f"summary{suffix}.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary["gates"], indent=1))


if __name__ == "__main__":
    main()
