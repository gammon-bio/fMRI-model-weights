"""BrainLM Stage 0c (plan: docs/brainlm_stage0c_plan.md, d37c86b + Amendment 1 9eb5217).

Synthetic plumbing only. Per seed: paired confound-on/off data -> cache layer-9 inputs (on: all
subjects; off: train + test_balanced) -> fine-tune vanilla + group-balanced reference (Stage 0b
settings) -> ground-truth site subspaces S1 (R1, head input) and S2 (R2, every reader input downstream
of layer 9, estimated in sequence) -> scripted edits H1, H2, L, K-a, K-e and the dose-response nulls
-> delta test_balanced AUC, delta in-distribution test AUC, m (effective input, gated) and m (literal R1).

  .venv/bin/python scripts/brainlm_stage0c.py run          # seeds 220-222, 3 h cap checked after seed 1
  .venv/bin/python scripts/brainlm_stage0c.py summarize    # gates from per_seed.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG0C = yaml.safe_load((ROOT / "configs" / "brainlm_stage0c.yaml").read_text())
N_THREADS = int(CFG0C["compute"]["n_threads"])
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = str(N_THREADS)

import numpy as np
import torch

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import brainlm_stage0b as B  # reuses make_cfg (Stage 0b data / fine-tune settings, unchanged)
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from fmritest.baselines.leace import LeaceEraser
from fmritest.brainlm import carriers as C
from fmritest.brainlm import model as M
from fmritest.brainlm import pack as P
from fmritest.brainlm import subspace as SS
from fmritest.data.generator import generate
from fmritest.data.splits import make_splits

torch.set_num_threads(N_THREADS)
AMP = bool(CFG0C["compute"]["bf16_autocast"])
OUT = ROOT / "results" / "brainlm_stage0c"
CFG0B = B.CFG0B


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auc(y, s) -> float:
    return float(roc_auc_score(y, s))


# --------------------------------------------------------------------------- reader capture
@torch.no_grad()
def capture(model: M.HeadModel, H: torch.Tensor, readers, batch_size: int = 32) -> dict:
    """Per-subject reader inputs: LN outputs averaged over the data tokens (R2), and the pooled
    final-LN output R1 ('head'). Returns {reader: (n, 768) float64}."""
    model.eval()
    mods = {"layer10.attn": model.layers[0].layernorm_before, "layer10.mlp": model.layers[0].layernorm_after,
            "layer11.attn": model.layers[1].layernorm_before, "layer11.mlp": model.layers[1].layernorm_after}
    idx = model.token_idx
    store = {r: [] for r in readers}
    hooks = []
    for r in readers:
        if r != "head":
            hooks.append(mods[r].register_forward_hook(
                lambda m, i, o, r=r: store[r].append(o.float()[:, idx].mean(1).double())))
    try:
        with M._amp(AMP):
            for i in range(0, len(H), batch_size):
                f = model.features(H[i:i + batch_size]).float().double()
                if "head" in store:
                    store["head"].append(f)
    finally:
        for h in hooks:
            h.remove()
    return {r: torch.cat(v).numpy() for r, v in store.items()}


def reader_linears(model: M.HeadModel, reader: str) -> list:
    if reader == "head":
        return [model.head]
    lay = model.layers[0 if reader.startswith("layer10") else 1]
    if reader.endswith("attn"):
        a = lay.attention.attention
        return [a.query, a.key, a.value]
    return [lay.intermediate.dense]


@torch.no_grad()
def fold_reader(model: M.HeadModel, reader: str, Q: np.ndarray) -> None:
    for lin in reader_linears(model, reader):
        W = lin.weight.data.double().numpy()
        lin.weight.data.copy_(torch.from_numpy(SS.fold_projection(W, Q)).to(lin.weight.dtype))


def head_params(model: M.HeadModel):
    return model.head.weight.data.double().numpy().copy(), model.head.bias.data.double().numpy().copy()


# --------------------------------------------------------------------------- evaluation of an edit
class Scorer:
    """Scores an edit from its R1 activations (test_balanced on / off, test on), head (w, b) and the
    effective-input transform T (A1.1): logits = R1 w^T + b; D_lit = R1_on - R1_off on test_balanced;
    D_eff = D_lit T^T. m = 1 - ||D Q_S1||^2 / ||D0 Q_S1||^2 with D0 from the unedited reference."""

    def __init__(self, y_tb, y_te, Q_S1, ref: dict):
        self.y_tb, self.y_te, self.Q = y_tb, y_te, Q_S1
        self.D0 = ref["tb_on"] - ref["tb_off"]
        self.auc_tb0 = auc(y_tb, ref["tb_on"] @ ref["w"].T[:, 0] + ref["b"][0])
        self.auc_te0 = auc(y_te, ref["te_on"] @ ref["w"].T[:, 0] + ref["b"][0])

    def score(self, tb_on, tb_off, te_on, w, b, T=None) -> dict:
        s_tb = tb_on @ w[0] + b[0]
        s_te = te_on @ w[0] + b[0]
        D = tb_on - tb_off
        De = D if T is None else D @ T.T
        a_tb, a_te = auc(self.y_tb, s_tb), auc(self.y_te, s_te)
        return {"test_balanced_auc": a_tb, "test_auc": a_te, "delta_test_balanced_auc": a_tb - self.auc_tb0,
                "delta_test_auc": a_te - self.auc_te0, "m": SS.metric_m(De, self.D0, self.Q),
                "m_literal": SS.metric_m(D, self.D0, self.Q)}


# --------------------------------------------------------------------------- one seed
def run_seed(seed: int, enc) -> dict:
    T = {}
    t_seed = time.time()
    cfg_on = B.make_cfg(float(CFG0C["effect"]))
    cfg_off = copy.deepcopy(cfg_on)
    cfg_off["data"]["confound"]["strength"] = 0.0
    held = seed % cfg_on["data"]["n_sites"]
    t = time.time()
    on = generate(cfg_on["data"], seed, heldout_site=held)
    off = generate(cfg_off["data"], seed, heldout_site=held)
    paired = {k: bool(np.array_equal(getattr(on, k), getattr(off, k)))
              for k in ("signal", "noise", "diagnosis", "site", "group", "site_freqs", "site_patterns")}
    paired["off_confound_zero"] = not off.confound.any()
    paired["X_on_eq_signal_plus_confound_plus_noise"] = bool(np.array_equal(on.X, (on.signal + on.confound) + on.noise))
    paired["X_off_eq_signal_plus_noise"] = bool(np.array_equal(off.X, off.signal + off.noise))
    if not all(paired.values()):
        raise SystemExit(f"paired-data check FAILED for seed {seed}: {paired}")
    sp = make_splits(on.site, held, tuple(cfg_on["split"]["frac"]), seed, group=on.group, strata=on.diagnosis)
    T["generate_pair"] = time.time() - t
    y, site, freqs = on.diagnosis, on.site, on.site_freqs
    train_sites = [c for c in range(cfg_on["data"]["n_sites"]) if c != held]
    low_site = train_sites[int(np.argmin(cfg_on["data"]["diagnosis"]["train_prevalence"]))]
    tr, va, tb, te = sp["train"], sp["val"], sp["test_balanced"], sp["test"]
    log(f"seed {seed}: paired check PASS; heldout {held}, low-prev site {low_site}, freqs {np.round(freqs, 3).tolist()}")

    t = time.time()
    med, iqr = P.fit_robust_scaler(on.X[tr])                       # one scaler (on-version train) for both
    Xon = P.robust_scale(on.X, med, iqr).astype(np.float32)
    off_idx = np.concatenate([tr, tb])
    Xoff = P.robust_scale(off.X[off_idx], med, iqr).astype(np.float32)
    del on, off
    T["scale"] = time.time() - t
    idx = M.token_index(P.data_patch_mask(cfg_on["data"]["n_regions"], cfg_on["data"]["n_timepoints"]))
    t = time.time()
    H = M.encode_recordings(enc, Xon, amp=AMP)
    T["cache_on"] = time.time() - t
    t = time.time()
    Hoff_all = M.encode_recordings(enc, Xoff, amp=AMP)
    T["cache_off"] = time.time() - t
    Hoff_tr, Hoff_tb = Hoff_all[: len(tr)], Hoff_all[len(tr):]
    log(f"  caches: on {tuple(H.shape)} {T['cache_on']:.0f}s, off {tuple(Hoff_all.shape)} {T['cache_off']:.0f}s")

    # ---- fine-tune (Stage 0b settings)
    ft = dict(CFG0B["finetune"])
    res = {"seed": seed, "effect": CFG0C["effect"], "heldout_site": held, "low_prevalence_site": int(low_site),
           "site_freqs": freqs.tolist(), "paired_check": paired, "n": {k: len(v) for k, v in sp.items()},
           "models": {}}
    models = {}
    for variant in ("vanilla", "reweighted"):
        ref = variant == "reweighted"
        t = time.time()
        m, info = M.train_head(enc, H[tr], y[tr], np.ones(len(tr)), H[va], y[va], np.ones(len(va)), idx, ft,
                               seed + (50 if ref else 0), amp=AMP, log=log,
                               group_ids=(site[tr] * 2 + y[tr]) if ref else None)
        T[f"finetune_{variant}"] = time.time() - t
        r = {"train_info": info}
        for k in ("test_balanced", "test", "test_site", "val_balanced"):
            r[k] = {"auc": auc(y[sp[k]], M.predict(m, H[sp[k]], amp=AMP)), "n": len(sp[k])}
        res["models"][variant] = r
        models[variant] = m
        log(f"  {variant}: AUC test_bal {r['test_balanced']['auc']:.3f} test {r['test']['auc']:.3f} "
            f"test_site {r['test_site']['auc']:.3f}")
    van = models["vanilla"]
    T["prepare"] = sum(T[k] for k in ("generate_pair", "scale", "cache_on", "cache_off", "finetune_vanilla",
                                      "finetune_reweighted"))

    # ---- S1 from R1 on train (unedited vanilla)
    t = time.time()
    R1_tr_on = capture(van, H[tr], ["head"])["head"]
    R1_tr_off = capture(van, Hoff_tr, ["head"])["head"]
    Q1, k1, curve1 = SS.energy_basis(R1_tr_on - R1_tr_off, CFG0C["subspace"]["energy"], CFG0C["subspace"]["k_cap"])
    ref = {"tb_on": capture(van, H[tb], ["head"])["head"], "tb_off": capture(van, Hoff_tb, ["head"])["head"],
           "te_on": capture(van, H[te], ["head"])["head"]}
    ref["w"], ref["b"] = head_params(van)
    sc = Scorer(y[tb], y[te], Q1, ref)
    T["subspace_S1"] = time.time() - t
    sub = {"S1": {"k": k1, "energy_curve": curve1}}
    log(f"  S1: k={k1}, energy curve {np.round(curve1[:k1 + 2], 3).tolist()}; unedited test_bal {sc.auc_tb0:.3f}")
    edits = {}
    rng = np.random.default_rng([seed, 4242])
    alphas, nper, tol = CFG0C["dose"]["alphas"], CFG0C["dose"]["n_per_alpha"], CFG0C["dose"]["overlap_tol"]

    # ---- H1 and its dose nulls (head-only: R1 unchanged; effective input (I - P_W) R1)
    t = time.time()
    w0, b0 = ref["w"], ref["b"]

    def h1(Wb):
        Tm = np.eye(Wb.shape[0]) - Wb @ Wb.T
        return sc.score(ref["tb_on"], ref["tb_off"], ref["te_on"], SS.fold_projection(w0, Wb), b0, Tm)

    edits["H1"] = h1(Q1) | {"level": "H1", "alpha": 1.0, "overlap": 1.0}
    for a in alphas:
        for j in range(nper):
            Wb = SS.overlap_basis(rng, Q1, a, tol)
            edits[f"H1_a{a:.2f}_{j}"] = h1(Wb) | {"level": "H1", "alpha": a, "overlap": SS.overlap(Wb, Q1)}
    T["H1_all"] = time.time() - t

    # ---- L: LEACE on R1 with site labels from the train + val pool, folded into the head
    t = time.time()
    pool = np.concatenate([tr, va])
    R1_pool = capture(van, H[pool], ["head"])["head"]
    er = LeaceEraser().fit(R1_pool, site[pool])
    wL, bL = SS.fold_leace(w0, b0, er.proj, er.mu)
    edits["L"] = sc.score(ref["tb_on"], ref["tb_off"], ref["te_on"], wL, bL, np.eye(768) - er.proj) | {
        "level": "L", "leace_rank": int(er.rank)}
    T["L"] = time.time() - t

    # ---- H2: S2 per reader, estimated in sequence on train (each reader with all earlier readers edited)
    t = time.time()
    m2 = copy.deepcopy(van)
    S2 = {}
    for r in SS.READERS:
        a_on = capture(m2, H[tr], [r])[r]
        a_off = capture(m2, Hoff_tr, [r])[r]
        Q, k, curve = SS.energy_basis(a_on - a_off, CFG0C["subspace"]["energy"], CFG0C["subspace"]["k_cap"])
        S2[r] = Q
        sub[f"S2.{r}"] = {"k": k, "energy_curve": curve}
        fold_reader(m2, r, Q)
        log(f"  S2 {r}: k={k}, energy curve {np.round(curve[:k + 2], 3).tolist()}")
    T["subspace_S2"] = time.time() - t
    res["subspaces"] = sub

    def h2(bases: dict, base_model=None):
        mm = copy.deepcopy(van) if base_model is None else base_model
        if base_model is None:
            for r in SS.READERS:
                fold_reader(mm, r, bases[r])
        R = {"tb_on": capture(mm, H[tb], ["head"])["head"], "tb_off": capture(mm, Hoff_tb, ["head"])["head"],
             "te_on": capture(mm, H[te], ["head"])["head"]}
        # R1 here is the pooled final-LN output after the layer 10-11 edits; the head step is folded in
        # head.weight, so the effective input is (I - P_head) R1 (A1.1)
        w, b = head_params(mm)
        Tm = np.eye(768) - bases["head"] @ bases["head"].T
        return sc.score(R["tb_on"], R["tb_off"], R["te_on"], w, b, Tm)

    t = time.time()
    edits["H2"] = h2(S2, base_model=m2) | {"level": "H2", "alpha": 1.0, "overlap": 1.0}
    T["H2_one"] = time.time() - t
    t = time.time()
    for a in alphas:
        for j in range(nper):
            bases = {r: SS.overlap_basis(rng, S2[r], a, tol) for r in SS.READERS}   # fixed S2 (A1.2)
            ov = float(np.mean([SS.overlap(bases[r], S2[r]) for r in SS.READERS]))
            edits[f"H2_a{a:.2f}_{j}"] = h2(bases) | {"level": "H2", "alpha": a, "overlap": ov}
        log(f"  H2 nulls alpha {a}: done ({time.time() - t:.0f}s)")
    T["H2_nulls"] = time.time() - t
    log(f"  H1 d {edits['H1']['delta_test_balanced_auc']:+.3f} m {edits['H1']['m']:.2f} | "
        f"H2 d {edits['H2']['delta_test_balanced_auc']:+.3f} m {edits['H2']['m']:.2f} | "
        f"L d {edits['L']['delta_test_balanced_auc']:+.3f} m {edits['L']['m']:.2f}")

    # ---- K-a, K-e: patch-kernel edits, full forward (reference = unedited full forward)
    t = time.time()
    W0 = M.kernel(enc)
    Xtb_on, Xte_on = Xon[tb], Xon[te]
    Xtb_off = Xoff[len(tr):]

    def kfull(W):
        with M.edited_kernel(enc, W):
            R = {"tb_on": M.encode_recordings(enc, Xtb_on, amp=AMP), "tb_off": M.encode_recordings(enc, Xtb_off, amp=AMP),
                 "te_on": M.encode_recordings(enc, Xte_on, amp=AMP)}
        return {k: capture(van, v, ["head"])["head"] for k, v in R.items()}

    Rk0 = kfull(W0)
    Rk0["w"], Rk0["b"] = w0, b0
    sck = Scorer(y[tb], y[te], Q1, Rk0)
    res["kernel_reference"] = {"test_balanced_auc": sck.auc_tb0, "test_auc": sck.auc_te0}
    for name, Qk in (("K-a", C.carrier_basis(freqs)), ("K-e", C.band_basis(int(CFG0B["controls"]["band_kmin"])))):
        R = kfull(C.project_out(W0, Qk))
        edits[name] = sck.score(R["tb_on"], R["tb_off"], R["te_on"], w0, b0, None) | {"level": name, "dims": int(Qk.shape[1])}
        log(f"  {name}: d test_bal {edits[name]['delta_test_balanced_auc']:+.3f} m {edits[name]['m']:.2f}")
    T["K_edits"] = time.time() - t

    res["edits"] = edits
    T["controls"] = sum(T[k] for k in ("subspace_S1", "H1_all", "L", "subspace_S2", "H2_one", "H2_nulls", "K_edits"))
    T["seed_wall"] = time.time() - t_seed
    res["timing_s"] = T
    return res


# --------------------------------------------------------------------------- gates
def level_stats(r: dict, lev: str) -> dict:
    E = [e for e in r["edits"].values() if e.get("level") == lev]
    d1 = r["edits"][lev]["delta_test_balanced_auc"]
    d0 = np.array([e["delta_test_balanced_auc"] for e in E if e["alpha"] == 0.0])
    a = np.array([e["alpha"] for e in E])
    d = np.array([e["delta_test_balanced_auc"] for e in E])
    curve = {f"{al:.2f}": {"mean": float(d[a == al].mean()), "sd": float(d[a == al].std(ddof=1)) if (a == al).sum() > 1 else 0.0,
                          "m_mean": float(np.mean([e["m"] for e in E if e["alpha"] == al])), "n": int((a == al).sum())}
             for al in sorted(set(a))}
    return {"n_edits": len(E), "delta1": d1, "delta0_mean": float(d0.mean()), "delta0_max": float(d0.max()),
            "spearman_alpha_delta": float(spearmanr(a, d)[0]), "curve": curve}


def gates(R: list[dict]) -> dict:
    need = min(2, len(R))
    v_tb = np.array([r["models"]["vanilla"]["test_balanced"]["auc"] for r in R])
    v_te = np.array([r["models"]["vanilla"]["test"]["auc"] for r in R])
    r_tb = np.array([r["models"]["reweighted"]["test_balanced"]["auc"] for r in R])
    r_te = np.array([r["models"]["reweighted"]["test"]["auc"] for r in R])
    out = {"G0c_shortcut": {"mean_headroom": float(np.mean(r_tb - v_tb)), "headroom": (r_tb - v_tb).tolist(),
                            "mean_vanilla_gap": float(np.mean(v_te - v_tb)), "vanilla_gap": (v_te - v_tb).tolist(),
                            "reweighted_test_auc": r_te.tolist()}}
    g = out["G0c_shortcut"]
    g["pass"] = bool(g["mean_headroom"] >= 0.15 and g["mean_vanilla_gap"] >= 0.10 and np.all(r_te >= 0.70))
    for lev in ("H1", "H2"):
        st = [level_stats(r, lev) for r in R]
        d1 = np.array([s["delta1"] for s in st])
        d0m = np.array([s["delta0_mean"] for s in st])
        d0x = np.array([s["delta0_max"] for s in st])
        rho = np.array([s["spearman_alpha_delta"] for s in st])
        c = {"mean_d1_minus_d0mean": float(np.mean(d1 - d0m)), "d1_minus_d0mean": (d1 - d0m).tolist(),
             "d1": d1.tolist(), "d0_mean": d0m.tolist(), "d0_max": d0x.tolist(),
             "d1_gt_d0max": (d1 > d0x).tolist(), "d1_gt_0": (d1 > 0).tolist()}
        c["pass"] = bool(c["mean_d1_minus_d0mean"] >= 0.05 and np.sum(d1 > d0x) >= need and np.sum(d1 > 0) >= need)
        out[f"G0c_control_{lev}"] = c
        out[f"G0c_dose_{lev}"] = {"spearman": rho.tolist(), "pass": bool(np.sum(rho >= 0.5) >= need),
                                  "curves": [s["curve"] for s in st]}
    sp_eff, sp_lit, n_ed = [], [], []
    for r in R:
        E = list(r["edits"].values())
        d = [e["delta_test_balanced_auc"] for e in E]
        sp_eff.append(float(spearmanr([e["m"] for e in E], d)[0]))
        sp_lit.append(float(spearmanr([e["m_literal"] for e in E], d)[0]))
        n_ed.append(len(E))
    out["G0c_metric"] = {"spearman_m_effective": sp_eff, "n_edits": n_ed,
                         "pass": bool(np.sum(np.array(sp_eff) >= 0.5) >= need),
                         "spearman_m_literal_not_gated": sp_lit}
    secs = [r["timing_s"]["prepare"] + r["timing_s"]["controls"] for r in R]
    out["G0c_compute"] = {"seconds_per_seed_prepare_plus_controls": secs, "seed_wall": [r["timing_s"]["seed_wall"] for r in R],
                          "mean": float(np.mean(secs)), "pass": True}
    return out


def cmd_summarize():
    R = json.loads((OUT / "per_seed.json").read_text())
    summary = {"plan": "docs/brainlm_stage0c_plan.md (d37c86b + Amendment 1 9eb5217)", "seeds": [r["seed"] for r in R],
               "gates": gates(R), "note": "Synthetic plumbing only; not a scientific result."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    g = summary["gates"]
    log(json.dumps({k: v.get("pass") for k, v in g.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "summarize":
        return cmd_summarize()
    t0 = time.time()
    enc = M.load_encoder(CFG0B["weights_dir"])
    log(f"encoder loaded + sha256 verified; threads {torch.get_num_threads()}; bf16 {AMP}")
    seeds = CFG0C["seeds"]
    per_seed = []
    for i, seed in enumerate(seeds):
        per_seed.append(run_seed(seed, enc))
        (OUT / "per_seed.json").write_text(json.dumps(per_seed, indent=1))
        if i == 0:
            proj = (time.time() - t0) * len(seeds)
            log(f"EXTRAPOLATION after seed {seed}: {time.time() - t0:.0f}s per seed -> about {proj:.0f}s total "
                f"(cap {CFG0C['wall_cap_s']}s)")
            if proj > CFG0C["wall_cap_s"]:
                log("projected total exceeds the cap: stopping as the plan requires")
                break
        log(f"seed {seed} done in {per_seed[-1]['timing_s']['seed_wall']:.0f}s")
    cmd_summarize()


if __name__ == "__main__":
    main()
