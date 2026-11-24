# BrainLM Stage 0c: ground-truth site subspace as the repair lever (2026-11-10)

> Not a scientific result. Synthetic plumbing for the Stage 1 design.

**Plan.** `docs/brainlm_stage0c_plan.md`: d37c86b plus Amendment 1 (9eb5217, sha256 618bb7bd...),
fixed before any seed. Seeds 220, 221 and 222; effect -0.3; Stage 0b data and fine-tune settings.
Nothing was changed after any seed ran. The gates below are computed by
`scripts/brainlm_stage0c.py summarize` from `results/brainlm_stage0c/per_seed.json`. I re-added the
main margins by hand from the per-seed values.

## 1. Gates

| gate | criterion | measured (seeds 220 / 221 / 222) | result |
|---|---|---|---|
| **G0c-shortcut** | mean headroom >= 0.15; mean vanilla gap >= 0.10; reweighted test AUC >= 0.70 every seed | headroom 0.217 / 0.046 / 0.253, mean **0.172**; gap 0.203 / 0.203 / 0.234, mean **0.213**; reweighted test 0.965 / 0.818 / 0.853 | **PASS** |
| **G0c-control H1** | mean [d(1) - mean d(0)] >= 0.05; d(1) > max d(0) in >= 2/3; d(1) > 0 in >= 2/3 | d(1) -0.059 / -0.188 / +0.027, mean margin **-0.073**; 1/3; 1/3 | **FAIL** |
| **G0c-dose H1** | Spearman(alpha, d) >= 0.5 in >= 2/3 seeds | 0.21 / -0.93 / 0.95 (1/3) | **FAIL** |
| **G0c-control H2** | as above | d(1) -0.108 / +0.181 / +0.317, mean margin **+0.131**; 2/3; 2/3 | **PASS** |
| **G0c-dose H2** | as above | 0.66 / 0.97 / 0.86 (3/3) | **PASS** |
| **G0c-metric** (diagnostic) | Spearman(m, d) over the 85 edits >= 0.5 in >= 2/3 seeds; m on the effective input (A1.1) | 0.53 / 0.24 / 0.88 (2/3) | **PASS** (seed 220 only 0.03 above the threshold) |
| (m literal, not gated) | | 0.51 / 0.85 / 0.29 | -- |
| **G0c-compute** | measured s/seed (prepare + controls) | 2633 / 2343 / 2296 s, mean 2424 s | measured |

d = change in vanilla test_balanced AUC caused by the edit. d(1) is the edit with alpha = 1, i.e. H1
or H2 itself. d(0) is over the 10 alpha = 0 nulls.

**Plan §4 reading.** G0c-shortcut passes, and G0c-control and G0c-dose both pass at **one level,
H2** (every reader downstream of layer 9). The plan's fixed interpretation therefore applies: *"Stage 1
can move to real data with that level as the mechanism lever"*. Caveats that this reading does not
remove (all exploratory, n = 3):

1. **H2 at alpha = 1 hurts in seed 220.**
   - d(1) = -0.108 and in-distribution change -0.267, while alpha = 0.75 gives +0.023.
   - H2 passes control on the mean and on 2/3 seeds, not on every seed.
2. **The alpha = 0 null is nearly inert.**
   - Random k-dim directions orthogonal to S in 768 dims change AUC by at most 0.001, with sd about
     0.001.
   - So the condition "d(1) > max d(0)" in practice reduces to "d(1) > 0". The control is weaker than
     its wording suggests.
   - Dose-response does the discriminating work: d rises with alpha in every seed at H2.
3. **The full projection costs diagnosis signal at R1.**
   - H1 removes S1 (k = 1 or 2) from the head. It drops in-distribution AUC by 0.26-0.29 in all
     seeds, and test_balanced AUC in 2 of 3.
   - In seeds 220 and 221 the H1 curve turns over between alpha = 0.75 and 1.
   - So at the head input, the dominant on-off direction is shared with the readout's diagnostic
     direction. Repair at R1 by subspace removal is not usable here. Repair spread over the
     upstream readers (H2) works in 2 of 3 seeds.
4. **m saturates and does not separate edits that matter.**
   - K-a and K-e both score m = 1.000, because removing the carrier bins at the input sets the
     confound difference D to about 0 everywhere. Yet K-e outperforms K-a in 2/3 seeds.
   - H1 has m = 1.000 while d is -0.19 to +0.03.
   - The G0c-metric pass rests on seed 222 (0.88) and a marginal seed 220 (0.53).
   - As the plan says, m alone is not ready to classify agent edits.
5. **Seed 221's headroom is only 0.046**, so the shortcut gate passes on the mean.

## 2. Paired data and subspaces

**Paired data.** Each seed is generated twice, the second time with `data.confound.strength` = 0.
In all 3 seeds:
- signal, noise, diagnosis, site, group, site_freqs and site_patterns are bit-identical;
- the off-version confound is exactly 0;
- X_on = (signal + confound) + noise exactly.

X_on - (X_off + confound) is at most 1.8e-15, because the additions happen in a different order. The
script stops on any mismatch.

**k per subspace.** Uncentred PCA of D over train subjects; smallest k reaching 90% of the energy,
capped at 8. Energy is the cumulative share at that k.

| subspace | seed 220 | seed 221 | seed 222 |
|---|---|---|---|
| S1 (R1, head input) | k=1 (0.938) | k=2 (0.945) | k=1 (0.909) |
| S2 layer10 attention input | 2 (0.944) | 3 (0.954) | 3 (0.951) |
| S2 layer10 MLP input | 2 (0.926) | 3 (0.939) | 3 (0.939) |
| S2 layer11 attention input | 2 (0.917) | 3 (0.937) | 3 (0.930) |
| S2 layer11 MLP input | 3 (0.947) | 3 (0.901) | 4 (0.965) |
| S2 head (after the layer 10-11 edits) | 3 (0.915) | 4 (0.916) | 4 (0.945) |

**S1 energy curve, first 8 PCs:**
- 220: 0.938, 0.991, 0.994, 0.996, 0.998, 0.998, 0.998, 0.999
- 221: 0.886, 0.945, 0.986, 0.994, 0.995, 0.996, 0.997, 0.998
- 222: 0.909, 0.972, 0.988, 0.993, 0.996, 0.997, 0.997, 0.998

The full curves for every subspace are in `per_seed.json` under `subspaces`. At R1 the confound
difference is close to rank 1-2. Further upstream it spreads over 2-4 dims.

## 3. Heads

AUC, vanilla / group-balanced reference.

| seed (held-out) | test_balanced | test | test_site | val_balanced |
|---|---|---|---|---|
| 220 (0) | 0.700 / 0.918 | 0.903 / 0.965 | 0.853 / 0.955 | 0.709 / 0.937 |
| 221 (1) | 0.702 / 0.748 | 0.905 / 0.818 | 0.770 / 0.850 | 0.685 / 0.754 |
| 222 (2) | 0.613 / 0.865 | 0.847 / 0.853 | 0.746 / 0.960 | 0.659 / 0.887 |

## 4. Edits

Each cell is d test_balanced AUC / d in-distribution test AUC / m (effective) / m literal. Off-target
damage is reported as the in-distribution change only, because no network readout applies to BrainLM
(A1.3).

| edit | seed 220 | seed 221 | seed 222 |
|---|---|---|---|
| H1 (head, S1) | -0.059 / -0.293 / 1.00 / 0.00 | -0.188 / -0.263 / 1.00 / 0.00 | +0.027 / -0.291 / 1.00 / 0.00 |
| H2 (all readers, S2 in sequence) | -0.108 / -0.267 / 1.00 / 0.99 | **+0.181** / +0.032 / 1.00 / 0.99 | **+0.317** / +0.066 / 1.00 / 0.99 |
| L (LEACE-site on R1, train+val pool, rank 2) | +0.117 / -0.141 / 0.87 / 0.00 | +0.079 / -0.091 / 0.63 / 0.00 | +0.112 / -0.195 / 0.94 / 0.00 |
| K-a (patch kernel, 4 carriers) | +0.099 / -0.091 / 1.00 / 1.00 | +0.066 / -0.093 / 1.00 / 1.00 | +0.118 / -0.177 / 1.00 / 1.00 |
| K-e (patch kernel, low-pass k >= 2) | +0.073 / -0.126 / 1.00 / 1.00 | +0.080 / -0.060 / 1.00 / 1.00 | +0.145 / -0.178 / 1.00 / 1.00 |

K-a and K-e are measured against the unedited full forward. Its test_balanced AUC (0.700 / 0.702 /
0.613) equals the cache path's in every seed.

**Comparisons** (reported, not gated):
- H2 beats L, K-a and K-e in seeds 221 and 222, by 0.10-0.20. It is the worst of the five in seed 220.
- L is the most consistent positive edit (+0.08 to +0.12), but it costs 0.09-0.20 of
  in-distribution AUC.
- H2 is the only edit that raises in-distribution AUC (seeds 221 and 222).
- The generic filter K-e is within 0.03 of K-a in every seed.

## 5. Dose curves

Each cell is mean d test_balanced AUC over the 10 nulls, (sd), with the mean m. alpha = 1 is the
single scripted edit. Overlaps were checked at construction to within 1e-6. The mean realised overlap
is 0.000000 / 0.250000 / 0.500000 / 0.750000.

**H1:**

| alpha | seed 220 | seed 221 | seed 222 |
|---|---|---|---|
| 0 | -0.000 (0.000), m 0.00 | +0.000 (0.000), m 0.00 | -0.000 (0.000), m 0.00 |
| 0.25 | -0.008 (0.001), m 0.44 | -0.003 (0.001), m 0.44 | +0.005 (0.002), m 0.44 |
| 0.5 | -0.007 (0.003), m 0.75 | -0.006 (0.004), m 0.75 | +0.012 (0.004), m 0.75 |
| 0.75 | +0.019 (0.001), m 0.94 | -0.024 (0.005), m 0.94 | +0.031 (0.006), m 0.94 |
| 1 | -0.059, m 1.00 | -0.188, m 1.00 | +0.027, m 1.00 |
| Spearman | 0.21 | -0.93 | 0.95 |

**H2:**

| alpha | seed 220 | seed 221 | seed 222 |
|---|---|---|---|
| 0 | -0.000 (0.001), m 0.03 | -0.001 (0.001), m 0.01 | -0.001 (0.001), m 0.03 |
| 0.25 | +0.011 (0.002), m 0.61 | +0.011 (0.007), m 0.60 | -0.002 (0.005), m 0.58 |
| 0.5 | +0.019 (0.002), m 0.89 | +0.041 (0.011), m 0.88 | +0.014 (0.007), m 0.87 |
| 0.75 | +0.023 (0.015), m 0.99 | +0.099 (0.022), m 0.98 | +0.046 (0.009), m 0.98 |
| 1 | -0.108, m 1.00 | +0.181, m 1.00 | +0.317, m 1.00 |
| Spearman | 0.66 | 0.97 | 0.86 |

At H2, d rises with alpha in every seed up to 0.75. Seed 220 turns over at alpha = 1.

## 6. Compute (measured; CPU, 2 threads, bf16)

| stage | seed 220 / 221 / 222 (s) |
|---|---|
| generate on + off | 15 / 12 / 13 |
| cache on (900 scans) | 204 / 146 / 157 |
| cache off (train + test_balanced, 480 scans) | 101 / 78 / 82 |
| fine-tune vanilla / reference (10 epochs each) | 418 + 411 / 407 + 450 / 407 + 388 |
| **prepare** | **1152 / 1097 / 1051** |
| S1 + references | 43 / 48 / 37 |
| H1 + 40 nulls (head only) | < 1 |
| L | 15 / 16 / 13 |
| S2 sequence (5 readers) | 111 / 119 / 102 |
| H2 + 40 nulls (about 19-25 s per edit) | 1000 / 774 / 780 |
| K-a + K-e + kernel reference (3 full forwards of 510 scans) | 312 / 288 / 314 |
| **controls** | **1482 / 1246 / 1245** |
| **prepare + controls** | **2633 / 2343 / 2296** |
| seed wall | 2675 / 2387 / 2339 |

**Total wall time:** 02:48:51 to 04:52:17, i.e. 2 h 03 min. After seed 220 the projected total was
8033 s, under the 10800 s cap. One heavy job ran at a time.

## 7. Deviations and notes

- **No deviations from the plan or Amendment 1.** Implementation choices are recorded as S15-S22 in
  `docs/decisions/0002-brainlm-stage0.md`.
- **The network readout does not apply (A1.3).** The Phase 2 masked-region-token readout has no
  counterpart in BrainLM's patch tokens.
- **Scoring path.** Edit AUCs are computed from captured R1 times the head weights in float64.
  - Model-level AUCs use the bf16 head (`predict`).
  - Here the two agree on unedited test_balanced to 3 decimals: 0.700 / 0.702 / 0.613.
  - Every delta is taken within one path.
- **New code:**
  - `scripts/brainlm_stage0c.py`;
  - `configs/brainlm_stage0c.yaml` (threads as a config value);
  - `fmritest/brainlm/subspace.py` and `tests/test_brainlm_subspace.py`, committed earlier with the
    amendment.
- **Earlier stages are unchanged.** No Stage 0 or 0b code changed, so their scripts still reproduce
  their results.
- **Nothing on disk.** No weights, heads or caches were saved. Logs are in the gitignored
  `outputs/brainlm_stage0c/`.

## 8. Review (main session, with a Fable check; 2026-11-10)

**Gates reproduced.** I recomputed every gate independently from `per_seed.json`, and so did the
Fable checker. Both match §1 exactly. No leakage is visible: S is estimated on train subjects only,
m is measured on test_balanced with S1 fixed, and L is fit on the agent-visible train+val pool. This
was checked at report level, not by a code audit. No weights or caches are on disk or in git. The
plan's locked rule applies: **Stage 1 may proceed with H2 as the lever.** What follows qualifies
that reading; it does not override it.

1. **The reference is not a ceiling, and H2 changes sign by seed.**
   - Edited vanilla test_balanced AUC vs the group-balanced reference: 0.592 vs 0.918, 0.883 vs
     0.748, and 0.930 vs 0.865.
   - Seed 221's reference is weak (headroom 0.046), so "H2 beats the reference" there says more
     about the reference than about H2.
   - In seed 220, H2 also drops in-distribution test AUC from 0.903 to 0.636.
2. **The dose pass and the control pass measure different things.**
   - The dose Spearman is 40 nulls plus one full edit, so by construction it hardly registers the
     full edit. H2's Spearman on the 40 nulls alone is 0.79 / 0.97 / 0.85; adding the full edit
     *lowers* seed 220's to 0.66.
   - Partial-overlap effects are small (at most +0.10). Most of the repair sits in the step from
     alpha 0.75 to 1: +0.023 -> -0.108, +0.099 -> +0.181, +0.046 -> +0.317.
   - In seed 220 all ten alpha = 0.75 nulls lie in -0.007 to +0.049. The full edit is a
     discontinuity, not an extrapolation of the curve.
3. **m is not usable as a cross-location mechanism measure** (stronger than caveat 4 in §1).
   - Over the five real edits (H1, H2, L, K-a, K-e), Spearman(m, d) is negative in every seed:
     -0.40 / -0.50 / -0.10.
   - The pooled 85-edit pass comes entirely from the nulls' alpha ordering; m is already 0.98-0.99
     at alpha = 0.75.
   - The nominal G0c-metric pass must not be cited as validating m.
4. **The site subspace overlaps the diagnostic readout, by an amount that varies with seed.**
   - H1 costs 0.26-0.29 of in-distribution AUC in every seed, and H2 costs 0.27 in seed 220.
   - The locked plan reports off-target damage but does not gate it.

**Required before the Stage 1 lock** (docs/decision_log.md item 20):
- (a) replace m with a measure that does not saturate and that ranks the real edits;
- (b) gate off-target in-distribution damage for the H2 lever;
- (c) treat the reweighted reference as a comparator, not as a ceiling on headroom;
- (d) power Stage 1 for seed-level sign flips at the full edit, and do not reuse the 40-null
  dose-Spearman shape as a test of the full edit.

**Corrections to the builder's text:** none. Its caveats 1-5 are accurate. Point 3 above
strengthens caveat 4.
