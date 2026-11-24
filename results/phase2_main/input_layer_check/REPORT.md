# Input-layer check (exploratory, post-unblinding, NOT pre-registered)

Script: `scripts/phase2_input_layer_check.py`. Inputs: the 30 included main-study fix seeds, run read-only
(no ledger writes, no score.json changed). Outputs: `per_episode.json`, `summary.json`, `transplant.json`.
Run 1 had a bug: carrier capture was keyed by site role, so the two high-prevalence sites overwrote each
other. Its outputs are kept unchanged in `../input_layer_check_run1_buggy_capture/`. Sections A, C and D
are numerically identical between the runs.

**Carrier subspace C.** In the 48-TR input window, C is spanned by sin and cos at the 4 planted site
frequencies (ground truth, never shown to agents), giving 8 of 48 dims.

**Sanity checks.**
- The unedited test_balanced AUC recomputed here equals the harness comparator in every seed (max
  difference 0.0).
- Re-applying each agent's embed edit as a pure projection reproduces its scored AUC to within 0.002 on
  average.

> Synthetic data only. The scripted edit uses ground-truth frequencies: it is a positive control and an
> upper bound for this edit family, not a fair competitor to the agents.

## A. Is the input embedding the lever? Yes

Scripted edit, no agent: `W_embed <- W_embed (I - P_C)`, on all 30 seeds.

| edit (embed.weight) | mean change in test_balanced AUC | seeds improved |
|---|---|---|
| project out all 4 site carriers (8 dims) | **+0.155** (95% CI 0.125 to 0.179; Wilcoxon p = 4e-8) | 28/30 |
| project out the low-prevalence site's carrier only (2 dims) | +0.117 | |
| project out both high-prevalence carriers (4 dims) | +0.042 | |
| project out the held-out site's carrier | -0.001 | |
| 50 random 8-dim subspaces (per-seed mean) | -0.007 | carrier edit above all 50 in the median seed |
| 4 off-carrier sinusoids (8 dims) | -0.001 | |

**Scripted 4-carrier edit vs comparators:**
- vs post-hoc pooled LEACE: +0.066;
- vs reweighted fine-tune: -0.030;
- vs the agents: +0.085.

**Off-target damage:** network readout change mean +0.013, minimum -0.021, against an allowed drop of
0.05.

**Failure mode: the low-prevalence carrier at 0.04 Hz** (11/30 seeds). 0.04 Hz lies inside the
HRF-filtered neural band.

| low-prevalence carrier | n | all 4 carriers | low-prev only | both high-prev |
|---|---|---|---|---|
| 0.040 Hz | 11 | +0.095 | +0.027 | +0.080 |
| 0.093 Hz | 10 | +0.191 | +0.167 | +0.028 |
| 0.147 Hz | 2 | +0.215 | +0.217 | +0.000 |
| 0.200 Hz | 7 | +0.179 | +0.159 | +0.016 |

All four seeds where the scripted edit gained less than 0.06 (013, 014, 026, 028) have the
low-prevalence carrier at 0.04 Hz.

## B. Did the agents' embed edits hit the planted mechanism? Yes, precisely

- **On target:** in 15 embed edits, the median share of delta energy inside C is **0.98** (chance 8/48 =
  0.17). In 14/15 it is above 0.85.
- **Not a dose effect:** alignment does not predict gain *within* the embed group (Spearman rho 0.20,
  p = 0.47), because nearly every edit is on target. The variance comes from which carrier was removed.

| agent strategy (carrier plane captured > 0.5) | n | mean agent change in AUC |
|---|---|---|
| low-prevalence carrier only | 8 | +0.181 |
| low-prevalence + one high-prevalence | 3 | +0.142 |
| both high-prevalence carriers | 4 | +0.073 |

- **Both-high-prevalence edits:** all 4 are in seeds where the low-prevalence carrier is at 0.04 Hz (003,
  006, 013, 031).
- **How the agents chose:** their notes show they picked frequencies by sweeping synthetic sinusoids
  through the weights and taking the peaks the logit responds to. In 0.04 Hz seeds those peaks are the
  high-prevalence carriers.
- **Choice vs best option:** relative to the better of the two scripted single strategies on the same
  seed, the choice was better in 006 and 031, worse in 003, and both options fail in 013. This is an
  interpretation of 4 episodes, not a test.

## C. Were embed editors handed easier seeds? No evidence

| group | n | unedited AUC | reweight fine-tune AUC | scripted 4-carrier gain on these seeds | agent gain |
|---|---|---|---|---|---|
| embed edit | 15 | 0.730 | 0.921 | +0.159 | **+0.144** |
| head / attention-output edit | 10 | 0.718 | 0.926 | +0.125 | **-0.007** |
| no-op | 5 | 0.742 | 0.861 | +0.202 | 0 |

Mann-Whitney, embed vs other: unedited AUC p = 0.76, reweight fine-tune AUC p = 0.76, scripted gain
p = 0.56. The non-embed agents' seeds were about as fixable, so the gap between groups is agent strategy,
not seed difficulty.

## D. Do agents' edits transplant to other seeds? Partly

Projecting each embed agent's input subspace out of every *other* seed's embed gives mean **+0.057**
(donor-mean 95% CI 0.039 to 0.075), with 60% of the 435 donor-recipient pairs improved.
- **Why only partly:** site-to-frequency assignment is permuted per seed. A transplant helps fully only
  when the donor's carrier is the recipient's low-prevalence carrier.
- **What carries over:** "a site carrier"; which carrier is seed-specific.

## Bottom line (exploratory)

The post-hoc split in the main report survives these checks:
- **The input embedding is a real lever.** A ground-truth carrier projection there recovers most of the
  reweighted fine-tune's gain without data.
- **The agents' embed edits removed planted site carriers almost exactly.** That is mechanism, not luck.
- **No seed selection was found** behind the embed vs other split.
- **The remaining gap** comes from agents not going to the input layer (10/30) or not editing at all
  (5/30), and from the 0.04 Hz seeds where the shortcut carrier overlaps the neural band.

A confirmatory test still needs a new pre-registered run on fresh seeds.
