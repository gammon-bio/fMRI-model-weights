# BrainLM Stage 0b: plan and gates (fixed before any gate seed is run)

**Date:** 2026-11-09. Approved by Caleb: "yes run stage 0b with your recommendations".

**Why.** Stage 0 failed G0-shortcut, and its positive control used a carrier set that was the same
in every seed (`docs/brainlm_stage0.md` §7). This run applies the fixes in
`docs/brainlm_study_design.md` §10, using synthetic data only.

> Not a scientific result. Synthetic plumbing for the Stage 1 design.

## 1. Changes from Stage 0

**C1. Carrier frequencies drawn per seed** (opt-in generator option; default behaviour unchanged).
- New key `data.confound.freq_choices`: a list of candidate frequencies. If present, the S site
  frequencies are drawn without replacement from it, using a **separate** RNG
  `np.random.default_rng([seed, 7919])`.
- The main stream still calls `rng.permutation(S)` exactly as now, so every other draw is
  unchanged.
- Candidates are the 16-sample DFT bins at TR 0.72 s, k = 2..7: 2/11.52 ... 7/11.52 Hz, i.e. 0.174,
  0.260, 0.347, 0.434, 0.521 and 0.608 Hz.
  - 4 of 6 are drawn per seed, giving 15 possible sets.
  - Bins are mutually orthogonal over a 16-sample patch.
- Requirement: with the key absent, the generator output is bit-identical to `generator-frozen-v2`.
  Verify this by hashing X, the labels and site_freqs for seeds 0, 1 and 200 before and after the
  change, and run the full existing test suite.

**C2. Balanced reference by group-balanced sampling, not importance weights.**
- Each batch samples the training site x label cells equally, with replacement.
- An "epoch" is len(train) samples.

**C3. Fixed epoch budget for both models, no early stopping.** E = 10 epochs; the final weights
are used.

**C4. Milder prevalence and more subjects.**
- `diagnosis.train_prevalence` = [0.8, 0.2, 0.8].
- `subjects_per_site` = 150, so 270 train subjects with minority cells of 12 per site.
- Balanced sets unchanged: 70 test_balanced and 30 val_balanced per training site.

**C5. Effect size chosen on separate tuning seeds.**
- Candidates: `diagnosis.effect` in {-0.4, -0.3}.
- Tuning seeds 300 and 301, which are never used for the gates.
- Rule: choose the effect with the larger mean val_balanced headroom (reweighted minus vanilla)
  over the two tuning seeds. A tie within 0.01 goes to -0.4.
- This is the **only** selection step. There is no adjustment after the gate seeds are run.

**C6. Unchanged:**
- 424 parcels, TR 0.72, 200 TRs, strength 0.6, spatial x spectral confound;
- K = 2 fine-tune of layers 10-11 + LN + linear head on the mean of the 351 data tokens;
- AdamW with the same learning rates;
- bf16 autocast;
- 2 threads.

## 2. Gate seeds and controls

**Gate seeds:** 210, 211, 212. They are fresh; no Stage 0 or tuning seed is reused.

Edits are applied to the patch kernel's temporal axis (all 3 channels and 16 rows). Each is scored
by a full forward of test_balanced through the vanilla model.

| id | edit | dims | needs ground truth |
|---|---|---|---|
| a | project out all 4 site carriers | 8 | yes (which bins) |
| b | low-prevalence site carrier only | 2 | yes |
| c | 20 random 8-dim subspaces **inside the band** (span of DFT bins k = 2..8, 13 dims) | 8 | no |
| d | the 2 unused candidate bins (in-band, off-carrier) | 4 | yes (complement) |
| e | generic low-pass: project out all 13 dims with k >= 2 (>= 0.174 Hz) | 13 | **no** |

For each edit, report:
- delta test_balanced AUC vs unedited;
- delta in-distribution test AUC;
- carrier share rho of the edit.

For c, also report the distribution of rho. That is the chance level for H-M1-BLM.

## 3. Gates (fixed now)

| gate | pass criterion |
|---|---|
| **G0b-shortcut** | mean over gate seeds of (reweighted - vanilla) test_balanced AUC >= 0.15, AND mean (vanilla test - vanilla test_balanced) >= 0.10, AND reweighted in-distribution test AUC >= 0.70 in every seed |
| **G0b-control** | mean [delta(a) - mean delta(c)] >= 0.05, AND delta(a) > max delta(c) in >= 2/3 seeds, AND delta(a) > 0 in >= 2/3 seeds |
| **G0b-specific** (diagnostic) | mean [delta(a) - delta(e)] >= 0.03, AND positive in >= 2/3 seeds |
| **G0b-compute** | measured seconds per seed for prepare + controls |

**Interpretation, fixed now:**
- **G0b-shortcut and G0b-control pass:** the Stage 1 regime is ready to move to real data.
- **G0b-specific fails:** a filter that needs no ground truth repairs as well as the carrier edit.
  - Stage 1 must then include the low-pass as a comparator, and frame H-M1-BLM as "agent edit vs
    generic filter".
  - Or Stage 1 must place carriers where a filter costs signal.
  - This is reported. It does not stop the next step.
- **G0b-shortcut fails:** Stage 1 does not proceed to real data. The cause is reported, and any
  further change is a new, separately approved stage.

## 4. Budget and boundaries

- Wall-time cap 4 h, single heavy job at a time, threads capped at 2 (a config value).
- After the first tuning seed, the build extrapolates the total. If it exceeds 4 h, it stops and
  reports.
- No weights, fine-tuned heads or caches on disk or in git. CC BY-NC-ND weights stay outside the
  repo.
- Outputs go to `results/brainlm_stage0b/` (Stage 0 results are untouched) and the report to
  `docs/brainlm_stage0b.md`.
