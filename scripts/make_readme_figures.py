"""Headline figures for the README, built only from committed result files (no model runs).

Usage: .venv/bin/python scripts/make_readme_figures.py
Writes PNG + PDF to results/figures_readme/.
  fig1_confirm_edit_class.*   Opus 5.5 confirmation study: change in balanced-test AUC by edit class
  fig2_fable_vs_opus.*        edit behaviour by model and by carrier stratum (exploratory)
  fig3_brainlm_stage0c_dose.* BrainLM Stage 0c dose curves for the two edit levels, per seed
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures_readme"
BLUE, RED, GREY = "#3498DB", "#E74C3C", "#95A5A6"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig1():
    M = json.loads((ROOT / "results/phase2_confirm/mechanism_episodes.json").read_text())
    order = [("ON", "carrier removed\nfrom input layer", RED), ("OFF", "other edit\n(mostly head refits)", BLUE),
             ("no_op", "no change", GREY)]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, (cls, label, col) in enumerate(order):
        d = np.array([m["delta_auc"] for m in M if m["edit_class"] == cls])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=46, color=col, alpha=0.85, edgecolor="white", lw=0.6)
        ax.hlines(d.mean(), i - 0.25, i + 0.25, color="black", lw=2)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ns = [sum(m["edit_class"] == c for m in M) for c, _, _ in order]
    ax.set_xticks(range(3), [f"{o[1]}\n(n = {n})" for o, n in zip(order, ns)])
    ax.set_ylabel("change in balanced-test AUC")
    ax.set_ylim(-0.12, 0.25)
    ax.set_title("Confirmation study (Opus 5.5, 27 audit-valid episodes)\n"
                 "ON vs OFF: +0.121 vs -0.005, one-sided MWU p = 0.004 (pre-registered H-M1)", fontsize=11)
    save(fig, "fig1_confirm_edit_class")


def fig2():
    rows = {"Opus 5.5": json.loads((ROOT / "results/phase2_confirm/mechanism_episodes.json").read_text()),
            "Fable 5.2": json.loads((ROOT / "results/fable52_pilot/mechanism_episodes.json").read_text())}
    classes = [("ON", "carrier removed (ON)", RED), ("OFF", "other edit (OFF)", BLUE), ("no_op", "no change", GREY)]
    groups = []
    for model, M in rows.items():
        for name, sel in [("0.04 Hz stratum", lambda m: abs(m["low_prev_freq"] - 0.04) < 1e-6),
                          ("other seeds", lambda m: abs(m["low_prev_freq"] - 0.04) >= 1e-6)]:
            S = [m for m in M if sel(m)]
            groups.append((f"{model}\n{name}\n(n = {len(S)})", [sum(m["edit_class"] == c for m in S) / len(S) for c, _, _ in classes]))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for j, (_, label, col) in enumerate(classes):
        v = np.array([g[1][j] for g in groups])
        ax.bar(x, v, bottom=bottom, color=col, width=0.62, label=label, edgecolor="white")
        bottom += v
    ax.set_xticks(x, [g[0] for g in groups], fontsize=10)
    ax.set_ylabel("fraction of episodes")
    ax.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    ax.set_title("Edit behaviour by model (exploratory; ON rate 4/10 vs 7/27, Fisher p = 0.44)", fontsize=11)
    save(fig, "fig2_fable_vs_opus")


def fig3():
    D = json.loads((ROOT / "results/brainlm_stage0c/per_seed.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    titles = {"H1": "H1: project site subspace out of the head input",
              "H2": "H2: project it out of every reader after layer 9"}
    for ax, lvl in zip(axes, ["H1", "H2"]):
        for s, col in zip(D, ["#1B4F72", "#2E86C1", "#85C1E9"]):
            E = [e for e in s["edits"].values() if e.get("level") == lvl]
            al = sorted({e["alpha"] for e in E})
            mu = [np.mean([e["delta_test_balanced_auc"] for e in E if e["alpha"] == a]) for a in al]
            ax.plot(al, mu, "-o", color=col, label=f"seed {s['seed']}")
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_title(titles[lvl], fontsize=10)
        ax.set_xlabel("overlap of the edit with the true site subspace")
    axes[0].set_ylabel("change in balanced-test AUC")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("BrainLM Stage 0c (synthetic plumbing): repair appears only at full alignment, and its sign varies by seed",
                 fontsize=10.5, y=1.04)
    save(fig, "fig3_brainlm_stage0c_dose")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
