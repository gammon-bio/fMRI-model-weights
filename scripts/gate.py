"""Scripted go/no-go gate. Criteria are pre-registered in configs/default.yaml (gate section).

Writes results/gate.json and results/gate.md. Exit code 0 = all gates pass, 2 = at least one fails.
A failing gate means: stop, report why, do not tune to make it pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fmritest.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    cfg = load_config(ap.parse_args().config)
    alpha, cond = cfg["gate"]["alpha"], cfg["gate"]["primary_condition"]
    rs = [json.loads((ROOT / "results" / "runs" / f"{cond}_seed{s}.json").read_text()) for s in cfg["seeds"]]

    def paired(a_, b_):
        a_, b_ = np.asarray(a_), np.asarray(b_)
        t = ttest_rel(a_, b_, alternative="less")
        return {"mean_diff": float((a_ - b_).mean()), "sd_diff": float((a_ - b_).std(ddof=1)),
                "t": float(t.statistic), "p_one_sided": float(t.pvalue), "n_better": int((a_ < b_).sum())}

    m = [r["recon"]["test"]["model"] for r in rs]
    best = [r["recon"]["test"]["best_linear"] for r in rs]
    g1 = {"model_mse": m, "best_linear_mse": best,
          "best_linear_name": [r["recon"]["test"]["best_linear_name"] for r in rs],
          **paired(m, best),
          "original_vs_pooled_fc": paired(m, [r["recon"]["test"]["fc"] for r in rs]),
          "criterion": "paired one-sided t, model < best linear baseline (pooled/lagged/per-site/per-site-lagged "
                       "FC) MSE on held-out subjects, p < alpha"}
    g1["pass"] = bool(g1["mean_diff"] < 0 and g1["p_one_sided"] < alpha)

    key = "model_" + {"masked_query_attention": "mqa"}[cfg["eval"]["network_readout"]] + "_test"
    sc = [r["network"][key]["score"] for r in rs]
    pn = [r["network"][key]["p_perm"] for r in rs]
    q95 = [r["network"][key]["null_q95"] for r in rs]
    umax = [r["network"]["untrained_mqa_null"]["max"] for r in rs]
    fcs = [r["network"]["fc_eig_varimax"]["score"] for r in rs]
    g2 = {"pass": bool(all(p < alpha for p in pn) and all(a_ > u for a_, u in zip(sc, umax))),
          "readout": key, "score": sc, "null_q95": q95, "p_perm": pn, "untrained_max": umax,
          "criterion": "beats shuffled-label null (p_perm < alpha) AND beats every random-init model, every seed",
          "context_fc_eig_varimax_score": fcs, "context_model_minus_fc": [a_ - f for a_, f in zip(sc, fcs)]}

    acc = [r["probe"]["model"]["acc"] for r in rs]
    pp = [r["probe"]["model"]["p_perm"] for r in rs]
    g3 = {"pass": bool(all(p < alpha for p in pp)), "acc": acc, "p_perm": pp,
          "chance": rs[0]["probe"]["model"]["chance"],
          "criterion": "subject-level site probe on embeddings beats permutation null in every seed",
          "context_untrained_acc": [r["probe"]["untrained"]["acc"] for r in rs]}

    prim = "model_" + cfg["eval"]["diag_features"]
    if rs[0].get("diagnosis"):
        id_ = [r["diagnosis"][prim]["test"]["bacc"] for r in rs]
        bal = [r["diagnosis"][prim]["test_balanced"]["bacc"] for r in rs]
        t4 = ttest_rel(bal, id_, alternative="less")
        gap = np.subtract(id_, bal)
        g4 = {"pass": bool(gap.mean() > 0 and t4.pvalue < alpha), "features": prim, "bacc_in_distribution": id_,
              "bacc_balanced": bal, "bacc_heldout_site": [r["diagnosis"][prim]["test_site"]["bacc"] for r in rs],
              "gap_mean": float(gap.mean()), "gap_sd": float(gap.std(ddof=1)), "p_one_sided": float(t4.pvalue),
              "criterion": "diagnosis probe balanced accuracy: in-distribution > correlation-broken (test_balanced), "
                           "paired one-sided t p < alpha"}
    else:
        g4 = {"pass": False, "criterion": "diagnosis task disabled in config"}

    # negative control (not a gate, but a failure here indicates leakage and invalidates G3)
    off = [json.loads((ROOT / "results" / "runs" / f"confound_off_seed{s}.json").read_text())
           for s in cfg["seeds"]] if "confound_off" in cfg["conditions"] else []
    ctrl = {"acc": [r["probe"]["model"]["acc"] for r in off], "p_perm": [r["probe"]["model"]["p_perm"] for r in off]}
    ctrl["n_significant"] = int(sum(p < alpha for p in ctrl["p_perm"]))
    ctrl["ok"] = ctrl["n_significant"] <= 1  # at alpha=.05 over 5 seeds, >1 false positive is suspicious

    gates = {"G1_reconstruction": g1, "G2_network_recovery": g2, "G3_site_decodable": g3, "G4_shortcut_measurable": g4,
             "negative_control_confound_off_site_probe": ctrl, "condition": cond, "alpha": alpha}
    allpass = g1["pass"] and g2["pass"] and g3["pass"] and g4["pass"] and ctrl["ok"]
    gates["ALL_PASS"] = bool(allpass)
    (ROOT / "results" / "gate.json").write_text(json.dumps(gates, indent=1))

    L = ["# Go/no-go gate (auto-generated by scripts/gate.py)", "", f"Condition: `{cond}`, alpha = {alpha}, "
         f"seeds = {cfg['seeds']}", "", "| gate | result | evidence |", "|---|---|---|",
         f"| G1 model beats best linear baseline on held-out reconstruction | {'PASS' if g1['pass'] else 'FAIL'} | "
         f"model - best linear MSE = {g1['mean_diff']:.4f} +/- {g1['sd_diff']:.4f} (SD over seeds), "
         f"p = {g1['p_one_sided']:.2g}, model better in {g1['n_better']}/{len(rs)} seeds; best linear = "
         f"{', '.join(sorted(set(g1['best_linear_name'])))} |",
         f"| (G1-original, v1 criterion) model beats pooled FC | "
         f"{'pass' if g1['original_vs_pooled_fc']['p_one_sided'] < alpha and g1['original_vs_pooled_fc']['mean_diff'] < 0 else 'fail'} | "
         f"diff {g1['original_vs_pooled_fc']['mean_diff']:.4f} +/- {g1['original_vs_pooled_fc']['sd_diff']:.4f}, "
         f"p = {g1['original_vs_pooled_fc']['p_one_sided']:.2g} |",
         f"| G2 planted networks recovered from internals | {'PASS' if g2['pass'] else 'FAIL'} | "
         f"score {np.mean(sc):.3f} +/- {np.std(sc, ddof=1):.3f}; shuffled-label q95 {np.mean(q95):.3f} (max p "
         f"{max(pn):.3g}); best untrained init {np.mean(umax):.3f} +/- {np.std(umax, ddof=1):.3f}; "
         f"FC eig+varimax {np.mean(fcs):.3f} (context, not gated) |",
         f"| G3 site decodable from embeddings | {'PASS' if g3['pass'] else 'FAIL'} | "
         f"acc {np.mean(acc):.3f} +/- {np.std(acc, ddof=1):.3f} vs chance {g3['chance']:.3f}; max p = {max(pp):.3g}; "
         f"untrained model acc {np.mean(g3['context_untrained_acc']):.3f} (shortcut is in the input) |",
         (f"| G4 diagnosis shortcut measurable | {'PASS' if g4['pass'] else 'FAIL'} | "
          f"bacc in-distribution {np.mean(g4['bacc_in_distribution']):.3f} vs correlation-broken "
          f"{np.mean(g4['bacc_balanced']):.3f} (gap {g4['gap_mean']:.3f} +/- {g4['gap_sd']:.3f}, p = "
          f"{g4['p_one_sided']:.2g}); held-out site {np.mean(g4['bacc_heldout_site']):.3f} |") if "gap_mean" in g4
         else f"| G4 diagnosis shortcut measurable | FAIL | {g4['criterion']} |",
         f"| Negative control: site probe with confound off | {'OK' if ctrl['ok'] else 'SUSPICIOUS'} | "
         f"acc {np.mean(ctrl['acc']) if off else float('nan'):.3f}; {ctrl['n_significant']}/{len(off)} seeds p < alpha |",
         "", f"**Overall: {'GO' if allpass else 'NO-GO'}**", ""]
    (ROOT / "results" / "gate.md").write_text("\n".join(L))
    print("\n".join(L))
    sys.exit(0 if allpass else 2)


if __name__ == "__main__":
    main()
