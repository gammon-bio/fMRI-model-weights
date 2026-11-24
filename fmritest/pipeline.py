"""One (condition, seed) run: generate -> split -> train -> evaluate model and baselines."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .baselines.fc import (components_from_affinity, fc_components, fc_reconstruct, group_fc, lagged_cov,
                           lagged_reconstruct)
from .baselines.leace import LeaceEraser
from .baselines.probes import fc_features, site_probe
from .data.generator import generate, zscore_subjects
from .data.splits import check_no_leakage, make_splits
from .eval.diagnosis import diag_eval, site_only_reference, subject_features
from .eval.metrics import network_recovery, recovery_with_null
from .eval.readouts import masked_query_attention, region_embedding_similarity, window_embeddings
from .interp.api import Inspector
from .models.train import eval_mse, make_eval_set, random_masks, to_windows, train_model, untrained_model


def _ids_hash(ids) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:12]


def run_seed(cfg: dict, seed: int, outdir: Path, log=print) -> dict:
    """cfg must already have condition overrides applied (fmritest.config.apply_overrides)."""
    t_start = time.time()
    cfg = copy.deepcopy(cfg)
    strength = cfg["data"]["confound"]["strength"]
    torch.set_num_threads(cfg["compute"]["n_threads"])
    outdir.mkdir(parents=True, exist_ok=True)
    mc, ec = cfg["model"], cfg["eval"]
    W, K = mc["window"], cfg["data"]["n_networks"]

    # ---------------------------------------------------------------- data & splits
    heldout = seed % cfg["data"]["n_sites"]
    ds = generate(cfg["data"], seed, heldout_site=heldout)
    X = zscore_subjects(ds.X)  # per-subject, per-region: no cross-subject statistics
    sp = make_splits(ds.site, heldout, tuple(cfg["split"]["frac"]), seed, group=ds.group, strata=ds.diagnosis)
    check_no_leakage(sp, ds.site, heldout)
    Xtr = X[sp["train"]]
    sets = {k: make_eval_set(X[sp[k]], W, ec["masks_per_window"], mc["mask_ratio"], seed + i)
            for i, k in enumerate(["val", "test", "test_site"])}
    # map eval-set row -> global subject id
    sets = {k: (x, m, sp[k][s]) for k, (x, m, s) in sets.items()}

    # ---------------------------------------------------------------- train
    log(f"[seed {seed} strength {strength}] train n={len(sp['train'])} heldout_site={heldout}")
    model, curves = train_model(Xtr, sets["val"], cfg, seed, log=log)
    torch.save(model.state_dict(), outdir / "model.pt")
    insp = Inspector(model, outdir / "interventions.jsonl", config=cfg, seed=seed)
    rand = untrained_model(cfg, X.shape[2], seed)
    insp_rand = Inspector(rand, None, config=cfg, seed=seed)
    site = ds.site

    # ---------------------------------------------------------------- reconstruction
    # Linear baselines (all fitted on training subjects only):
    #   fc            pooled instantaneous correlation (conditional Gaussian)  -- the spec'd FC baseline
    #   fc_lag        pooled, with +/- L timepoints of temporal context (the model also sees context)
    #   fc_site       per-site instantaneous covariance (uses the test subject's site label; seen sites only)
    #   fc_site_lag   per-site with temporal context
    C_train = group_fc(Xtr)
    Lg = ec["fc_lags"]
    S_lag = lagged_cov(Xtr, Lg)
    tr_sites = np.unique(site[sp["train"]])
    S_site = {c: (lagged_cov(X[sp["train"][site[sp["train"]] == c]], 0),
                  lagged_cov(X[sp["train"][site[sp["train"]] == c]], Lg)) for c in tr_sites}
    mse = lambda pred, x, m: float(((pred - x) ** 2 * m[..., None]).sum() / (m.sum() * W))
    recon = {}
    for k in ("test", "test_site"):
        x, m, subj = sets[k]
        r = {"model": eval_mse(model, x, m), "fc": mse(fc_reconstruct(C_train, x, m), x, m),
             "fc_lag": mse(lagged_reconstruct(S_lag, Lg, x, m), x, m),
             "zero": float((x ** 2 * m[..., None]).sum() / (m.sum() * W)), "untrained": eval_mse(rand, x, m)}
        if k == "test":  # site-conditional baselines exist only for sites seen in training
            p0, pL = x.copy(), x.copy()
            for c in tr_sites:
                sel = site[subj] == c
                p0[sel] = lagged_reconstruct(S_site[c][0], 0, x[sel], m[sel])
                pL[sel] = lagged_reconstruct(S_site[c][1], Lg, x[sel], m[sel])
            r["fc_site"], r["fc_site_lag"] = mse(p0, x, m), mse(pL, x, m)
        r["best_linear"] = min(v for kk, v in r.items() if kk.startswith("fc"))
        r["best_linear_name"] = min((v, kk) for kk, v in r.items() if kk.startswith("fc"))[1]
        recon[k] = r

    # ---------------------------------------------------------------- network recovery
    M = ds.membership
    xw_test = to_windows(X[sp["test"]], W).reshape(-1, X.shape[2], W).astype(np.float32)
    xw_site = to_windows(X[sp["test_site"]], W).reshape(-1, X.shape[2], W).astype(np.float32)
    meth = ec["network_extraction"]
    P = ec["n_perm_network"]
    net = {}
    A_test = masked_query_attention(insp, xw_test)
    net["model_mqa_test"] = recovery_with_null(components_from_affinity(A_test, K, meth), M, P, seed)
    net["model_mqa_test_site"] = recovery_with_null(
        components_from_affinity(masked_query_attention(insp, xw_site), K, meth), M, P, seed)
    for l in range(mc["n_layers"]):
        net[f"model_mqa_layer{l}"] = recovery_with_null(
            components_from_affinity(masked_query_attention(insp, xw_test, layers=[l]), K, meth), M, P, seed)
    net["model_region_emb"] = recovery_with_null(
        components_from_affinity(region_embedding_similarity(insp), K, meth), M, P, seed)
    net["untrained_mqa_test"] = recovery_with_null(
        components_from_affinity(masked_query_attention(insp_rand, xw_test), K, meth), M, P, seed)
    # untrained-model null: the masked-query readout inherits input FC even at random init (masking a
    # region removes its key from the softmax), so the trained model must beat random inits, not just
    # the shuffled-label null.
    unt = []
    for i in range(ec["n_untrained_inits"]):
        ins_i = Inspector(untrained_model(cfg, X.shape[2], seed, init=i), None)
        unt.append(network_recovery(components_from_affinity(masked_query_attention(ins_i, xw_test), K, meth), M)["score"])
    net["untrained_mqa_null"] = {"scores": unt, "max": float(max(unt)), "q95": float(np.quantile(unt, 0.95)),
                                 "model_minus_max": net["model_mqa_test"]["score"] - float(max(unt))}
    for m_ in ("eig_varimax", "pca", "ica"):
        net[f"fc_{m_}"] = recovery_with_null(fc_components(Xtr, K, m_, seed), M, P, seed)

    # ---------------------------------------------------------------- site decodability
    nw = X.shape[1] // W
    subj_tr = np.repeat(sp["train"], nw)
    subj_te = np.repeat(sp["test"], nw)
    xw_tr = to_windows(Xtr, W).reshape(-1, X.shape[2], W).astype(np.float32)
    pp = dict(C=ec["probe_C"], n_perm=ec["n_perm_probe"], seed=seed)
    probes = {}
    E_tr, E_te = window_embeddings(insp, xw_tr), window_embeddings(insp, xw_test)
    probes["model"] = site_probe(E_tr, subj_tr, E_te, subj_te, site, **pp)
    probes["untrained"] = site_probe(window_embeddings(insp_rand, xw_tr), subj_tr,
                                     window_embeddings(insp_rand, xw_test), subj_te, site, **pp)
    probes["fc"] = site_probe(fc_features(Xtr), sp["train"], fc_features(X[sp["test"]]), sp["test"], site, **pp)
    # LEACE on the pooled embedding (representation-level sanity check of the eraser)
    er = LeaceEraser().fit(E_tr, site[subj_tr])
    probes["model_leace_pooled"] = site_probe(er.transform(E_tr), subj_tr, er.transform(E_te), subj_te, site, **pp)

    # LEACE inside the model: per-token eraser on the final residual stream, fitted on train tokens from
    # both unmasked windows (probe distribution) and randomly masked windows (reconstruction distribution)
    hook = f"blocks.{mc['n_layers'] - 1}.hook_resid_post"
    m_tr = random_masks(np.random.default_rng(seed + 4242), len(xw_tr), X.shape[2], mc["mask_ratio"])
    T_tr = np.concatenate([window_embeddings(insp, xw_tr, pool=False),
                           window_embeddings(insp, xw_tr, pool=False, masks=m_tr)])  # (2n, R, d)
    tok_er = LeaceEraser().fit(T_tr.reshape(-1, T_tr.shape[-1]),
                               np.tile(np.repeat(site[subj_tr], X.shape[2]), 2))
    leace = {}
    diag_on = bool(cfg["data"].get("diagnosis", {}).get("enabled", False))
    with insp.intervention():
        insp.apply_affine(hook, tok_er.transform, label="LEACE token-level site eraser (fit on train)")
        for k in ("test", "test_site"):
            leace[f"recon_{k}"] = eval_mse(model, *sets[k][:2])
        leace["probe"] = site_probe(window_embeddings(insp, xw_tr), subj_tr,
                                    window_embeddings(insp, xw_test), subj_te, site, **pp)
        diag_leace_in_model = None
        if diag_on:
            Fl = {k: subject_features(insp, X[v], W, ec["diag_features"]) for k, v in sp.items()}
            diag_leace_in_model = diag_eval(Fl, ds.diagnosis, site, sp, ec["diag_probe_C"], seed)
    assert insp.is_pristine()
    leace["diagnosis"] = diag_leace_in_model

    # ---------------------------------------------------------------- planted diagnosis (shortcut task)
    diag = None
    if diag_on:
        y, dC = ds.diagnosis, ec["diag_probe_C"]
        feats = {
            "model_token_concat": {k: subject_features(insp, X[v], W, "token_concat") for k, v in sp.items()},
            "model_pooled": {k: subject_features(insp, X[v], W, "pooled") for k, v in sp.items()},
            "untrained_token_concat": {k: subject_features(insp_rand, X[v], W, "token_concat") for k, v in sp.items()},
            # secondary, exploratory (added after a pilot showed weak diagnosis signal in token features):
            "model_attention": {k: subject_features(insp, X[v], W, "attention") for k, v in sp.items()},
            "fc": {k: fc_features(X[v]) for k, v in sp.items()},
        }
        diag = {name: diag_eval(F, y, site, sp, dC, seed) for name, F in feats.items()}
        prim = "model_" + ec["diag_features"]
        diag[prim + "+reweight"] = diag_eval(feats[prim], y, site, sp, dC, seed, reweight=True)
        diag[prim + "+leace_site"] = diag_eval(feats[prim], y, site, sp, dC, seed, erase_site=True)
        diag["fc+reweight"] = diag_eval(feats["fc"], y, site, sp, dC, seed, reweight=True)
        diag["site_only"] = site_only_reference(y, site, sp, seed)
        diag["model_attention+reweight"] = diag_eval(feats["model_attention"], y, site, sp, dC, seed, reweight=True)

    res = {
        "seed": seed, "strength": strength, "heldout_site": int(heldout),
        "splits": {k: {"n": int(len(v)), "ids_sha": _ids_hash(v), "sites": sorted(set(site[v].tolist()))}
                   for k, v in sp.items()},
        "recon": recon, "network": net, "probe": probes, "leace_in_model": leace, "diagnosis": diag,
        "train_seconds": curves["seconds"], "total_seconds": time.time() - t_start,
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
    (outdir / "metrics.json").write_text(json.dumps(res, indent=1))
    (outdir / "curves.json").write_text(json.dumps(curves))
    np.save(outdir / "membership.npy", M)
    np.save(outdir / "mqa_test.npy", A_test)
    log(f"[seed {seed} strength {strength}] done in {res['total_seconds']:.0f}s | "
        f"MSE model {recon['test']['model']:.4f} fc {recon['test']['fc']:.4f} | "
        f"net {net['model_mqa_test']['score']:.3f} (null95 {net['model_mqa_test']['null_q95']:.3f}) | "
        f"site acc {probes['model']['acc']:.2f} p={probes['model']['p_perm']:.3f}"
        + (f" | diag bacc ID {diag['model_token_concat']['test']['bacc']:.2f} "
           f"balanced {diag['model_token_concat']['test_balanced']['bacc']:.2f} "
           f"site {diag['model_token_concat']['test_site']['bacc']:.2f}" if diag else ""))
    return res
