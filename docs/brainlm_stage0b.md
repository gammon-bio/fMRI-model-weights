# BrainLM Stage 0b: revised shortcut regime and specific controls (2026-11-09)

> Not a scientific result. Synthetic plumbing for the Stage 1 design.

**Plan.** `docs/brainlm_stage0b_plan.md` (fixed in 6bfab91, sha256 eba8a071...). Every gate and
control below follows it as written. The only selection step was C5, on tuning seeds 300 and 301.
Nothing was changed after the gate seeds (210-212) were run.

**Verdict.**

| gate | criterion | measured | result |
|---|---|---|---|
| **G0b-shortcut** | mean headroom >= 0.15; mean vanilla gap >= 0.10; reweighted test AUC >= 0.70 in every seed | headroom 0.167 / 0.163 / 0.237, mean **0.189**; gap 0.124 / 0.218 / 0.205, mean **0.182**; reweighted test 0.974 / 0.820 / 0.957 | **PASS** |
| **G0b-control** | mean [d(a) - mean d(c)] >= 0.05; d(a) > max d(c) in >= 2/3 seeds; d(a) > 0 in >= 2/3 seeds | +0.033 / +0.033 / -0.022, mean **+0.015**; d(a) > max d(c) in **1/3**; d(a) > 0 in 3/3 | **FAIL** |
| **G0b-specific** (diagnostic) | mean [d(a) - d(e)] >= 0.03; positive in >= 2/3 seeds | +0.068 / +0.059 / -0.036, mean **+0.0303**; positive in 2/3 | **PASS** (0.0003 above the threshold) |
| **G0b-compute** | measured s/seed for prepare + controls | 2536 / 2502 / 2437 s, mean **2492 s** | measured |

d(x) = change in vanilla test_balanced AUC under edit x, vs the unedited full forward.

**Reading, against the interpretation fixed in plan §3.**
- **The shortcut regime now works.**
  - Vanilla in-distribution AUC is clearly above its test_balanced AUC in every seed.
  - The group-balanced reference reaches test_balanced 0.82-0.96 with in-distribution AUC 0.82-0.97.
    The Stage 0 overcorrection is gone.
- **The positive control is not specific against a band-matched null.**
  - Projecting out the 4 true carriers (a) repairs: +0.10 to +0.15.
  - But a random 8-dim subspace of the same 13-dim band (c) repairs almost as much: mean +0.10 to
    +0.12, best +0.14 to +0.19.
  - (a) beat all 20 random subspaces only in seed 210. It was at the 95th percentile in seed 211 and
    the 30th in seed 212.
  - So on this synthetic data, most of the repair comes from removing in-band (>= 0.174 Hz)
    temporal content in general, not from the specific carrier bins.
- Plan §3 does not spell out an outcome for "shortcut passes, control fails". Its pass condition
  for moving to real data is that **both** pass. That condition is **not met**.
- G0b-specific passes by 0.0003, and fails in seed 212, where the 13-dim low-pass (+0.133) beats
  (a) (+0.097). A pass this thin is not evidence that the carrier edit is specific.

## 1. What changed from Stage 0 (C1-C5) and how it was built

- **C1, generator.**
  - New opt-in key `data.confound.freq_choices` (`fmritest/data/generator.py`). The S frequencies
    are drawn without replacement from the candidates by `np.random.default_rng([seed, 7919])`.
  - The main-stream `rng.permutation(S)` call is kept.
  - Candidates are the DFT bins k = 2..7: 0.174, 0.260, 0.347, 0.434, 0.521 and 0.608 Hz.
  - **Bit-identity check.** sha256 of (X, diagnosis, site, group, site_freqs) for seeds 0, 1 and
    200 with `configs/default.yaml` was taken before and after the edit, with the key absent. The
    hashes are identical:

    | seed | sha256 |
    |---|---|
    | 0 | bfa04baf... |
    | 1 | 10203c31... |
    | 200 | f6922437... |

    Before the edit, generator.py was identical to the tag `generator-frozen-v2`.
  - New test `tests/test_generator.py::test_freq_choices_opt_in` checks:
    - 4 distinct candidates are drawn;
    - the sets differ across seeds;
    - signal, noise and site patterns are unchanged.
  - The gate seeds drew 3 different carrier sets: {0.174, 0.608, 0.260, 0.521},
    {0.347, 0.260, 0.608, 0.521} and {0.347, 0.260, 0.434, 0.174}.
- **C2, `fmritest/brainlm/model.py`.** `train_head(..., group_ids=...)`:
  - each epoch draws len(train) indices;
  - for each, a site x label cell is picked uniformly, then a member uniformly, with replacement;
  - the loss is unweighted.
- **C3.** `finetune.early_stopping: false` runs 10 epochs and keeps the final weights. In this mode
  val loss is computed only after the last epoch, for reporting.
  - The Stage 0 paths (importance weights, early stopping) are unchanged and remain the default.
  - `scripts/brainlm_stage0.py` is untouched, so the Stage 0 numbers stay reproducible.
- **C4.** Prevalence [0.8, 0.2, 0.8], 150 subjects per site. Train has 270 subjects; the minority
  cells hold **18** per training site in train (6 in val).
  - The plan text says 12. 150 x 0.2 x 0.6 = 18; this is a typo in the plan's derived count, not a
    deviation in the run.
- **Config.** `configs/brainlm_stage0b.yaml`: threads 2 (a config value), bf16, all C1-C5 settings.
  The script is `scripts/brainlm_stage0b.py` with commands `tune`, `gate` and `summarize`.
- **Controls** (`fmritest/brainlm/carriers.py`):
  - `band_basis(2)` is the 13-dim span of the DFT bins k = 2..8 (Nyquist cos only).
  - `random_in_span` draws c from that span, using rng [seed, 99].
  - d is the 2 candidate bins not drawn for the seed.
  - e is the whole band.
  - `carrier_share` gives rho = ||dW Qc||^2 / ||dW||^2 on the temporal axis, as in Phase 2.
- Implementation choices are in ADR 0002, addendum S9-S14.

## 2. C5 tuning (seeds 300 and 301; fine-tune and val_balanced only, no controls)

val_balanced AUC, vanilla / reference, and headroom.

| seed (carrier set, Hz) | effect -0.4 | effect -0.3 |
|---|---|---|
| 300 (0.608, 0.174, 0.521, 0.434) | 0.791 / 1.000, **+0.209** | 0.717 / 0.840, **+0.123** |
| 301 (0.521, 0.260, 0.434, 0.347) | 0.925 / 0.993, **+0.068** | 0.681 / 0.957, **+0.276** |
| mean headroom | **0.139** | **0.200** |

**Rule (C5), applied mechanically by the script:** take the larger mean val_balanced headroom; a
tie within 0.01 goes to -0.4. Since 0.200 - 0.139 = 0.061 is not a tie, **effect = -0.3**. The
record is in `results/brainlm_stage0b/tuning.json` ("selection"). The `gate` command refuses any
effect other than the recorded one.

**Extrapolation after the first tuning seed:**
- 2047 s per tuning seed, so about 68 min for tuning.
- Plus about 3 x 2600 s for the gate seeds.
- That gives about 3.3 h, under the 4 h cap, so the run continued.

## 3. Gate seeds 210-212 (effect -0.3)

**Heads** (AUC, vanilla / group-balanced reference):

| seed (held-out, low-prev site) | test_balanced | test (in-dist) | test_site | val_balanced |
|---|---|---|---|---|
| 210 (2, 1) | 0.791 / 0.958 | 0.914 / 0.974 | 0.828 / 0.966 | 0.778 / 0.944 |
| 211 (3, 1) | 0.652 / 0.815 | 0.870 / 0.820 | 0.710 / 0.840 | 0.600 / 0.785 |
| 212 (0, 2) | 0.714 / 0.951 | 0.919 / 0.957 | 0.901 / 0.929 | 0.714 / 0.917 |

The unedited full forward reproduces the cache-path test_balanced AUC exactly in all seeds (0.791,
0.652, 0.714).

**Controls on vanilla.** Each edit is a full forward of test_balanced + test. Each cell gives
d test_balanced AUC / d test AUC / rho.

| edit (dims) | seed 210 | seed 211 | seed 212 |
|---|---|---|---|
| (a) all 4 carriers (8) | **+0.150** / +0.044 / 1.00 | **+0.138** / +0.038 / 1.00 | **+0.097** / -0.123 / 1.00 |
| (b) low-prev site carrier (2) | +0.026 / -0.002 / 1.00 | +0.078 / +0.050 / 1.00 | +0.002 / +0.002 / 1.00 |
| (c) 20 random in-band 8-dim, d test_bal mean (sd) [min, max] | +0.117 (0.018) [+0.080, +0.144] | +0.105 (0.023) [+0.053, +0.153] | +0.119 (0.033) [+0.053, +0.187] |
| (c) mean d test AUC | +0.018 | +0.023 | -0.028 |
| (c) rho: mean (sd), 5 / 50 / 95 % quantiles | 0.639 (0.061), 0.550 / 0.639 / 0.719 | 0.601 (0.059), 0.498 / 0.596 / 0.688 | 0.597 (0.082), 0.486 / 0.591 / 0.720 |
| (a) percentile among c | 100 | 95 | 30 |
| (d) 2 unused bins (4) | +0.061 / -0.003 / 0.00 | +0.032 / +0.021 / 0.00 | -0.002 / -0.026 / 0.00 |
| (e) low-pass k >= 2 (13) | +0.081 / -0.035 / 0.65 | +0.079 / -0.035 / 0.59 | +0.133 / -0.064 / 0.60 |

**Chance level for carrier share (H-M1-BLM).** rho of in-band random 8-dim edits ranges
0.48-0.73, with a median of about 0.60. It is above 8/13 = 0.62 in some seeds because rho weights
directions by the kernel's energy. A classification threshold of rho >= 0.5 would label almost
every random in-band edit "carrier". The design §10 suggestion (rho >= 0.8, or a percentile against
matched random edits) is needed.

**Other observations** (exploratory, n = 3):
- Removing the 2 bins that carry **no** site signal (d, rho 0) still helps in 2 of 3 seeds, by up
  to +0.06.
- In seed 212, (a) costs 0.12 of in-distribution AUC.
- Both observations point the same way as the main result: in-band temporal content in BrainLM's
  patch embedding carries a site-like signal beyond the planted carrier bins, or removing it
  regularises the vanilla readout. Stage 0b cannot tell these apart.

## 4. Compute (measured; CPU, 2 threads, bf16 autocast)

| stage | per seed (gate seeds 210 / 211 / 212) |
|---|---|
| generate (900 subjects) + scale | 7 s |
| cache layer-9 (900 scans) | 174 / 162 / 157 s |
| fine-tune vanilla, 10 epochs x 270 | 436 / 427 / 405 s |
| fine-tune reference, 10 epochs x 270 | 413 / 414 / 420 s |
| head evals | 42 / 46 / 40 s |
| **prepare** | **1029 / 1009 / 989 s** |
| one edited forward (300 scans: test_balanced + test) | 60.3 / 59.8 / 57.9 s mean |
| controls (unedited + 24 edits) | 1507 / 1493 / 1448 s |
| **prepare + controls** | **2536 / 2502 / 2437 s** |
| seed wall | 2578 / 2548 / 2478 s |

**Run time.**
- Tuning: 4 prepare runs, 4118 s (1015-1038 s each).
- Total Stage 0b wall time: 15:32-18:52, about 3.3 h, under the 4 h cap. There was a 1-minute gap
  between tuning and gating, caused by a broken chaining wrapper (below).
- One heavy job ran at a time.

**30-seed extrapolation** (formula estimate, not measured):
- Per edit, about 0.20 s/scan.
- N = 50 nulls on test_balanced only: 1010 + 58 x 42 s = about 3450 s/seed, so about 29 h.
- N = 200: about 9750 s/seed, so about 81 h.

## 5. Deviations and issues

- **None in the gates or controls.** Everything was computed as written in the plan, from
  `per_seed.json`. The arithmetic was recomputed by hand from the per-seed values:
  - control margin (0.033 + 0.033 - 0.022) / 3 = 0.015;
  - specific margin (0.068 + 0.059 - 0.036) / 3 = 0.030;
  - headroom 0.189.
- The plan's "minority cells of 12 per site" is 18 in train (see C4).
- `apply_overrides` cannot add the new key, so `freq_choices` is written into the config dict
  directly (ADR S9). `configs/default.yaml` is unchanged.
- An attempt to chain the gate run behind tuning with `pgrep -f` matched its own wrapper and never
  fired. It was killed and the gate run was started by hand, 54 s after tuning ended. Results are
  unaffected.
- `~/.claude/scripts/heavy-check.sh` is absent on this container. Load was checked with `uptime`.
- Full test suite: 81 passed, 1 failed right after C1; 84 passed, 1 failed at the end, with the new
  tests included. The one failure both times is the known `test_task_md_has_no_root_and_root_is_inferred`.
- No weights, heads or caches on disk. Logs are in the gitignored `outputs/brainlm_stage0b/`.

## Review notes (main session, 2026-11-09)

- **Gates recomputed independently from `per_seed.json`:** the shortcut, control and specific
  numbers all match.
- **Erratum to the plan.** `docs/brainlm_stage0b_plan.md` C4 says the minority cells are 12 per
  site. The real count is 18 (150 subjects x 0.6 train x 0.2). The run used the plan's settings, so
  only the count stated in the plan was wrong. The plan file stays unchanged because it is locked.
- **Dose-response: corrected after an independent Fable check.**
  - Among the 20 random in-band 8-dim edits, the Spearman correlation between overlap and change
    in AUC is 0.25 / 0.03 / -0.12. But random 8-of-13 subspaces all have overlap near 8/13
    (observed 0.51-0.71), so n = 20 has almost no power to detect a dose-response. The first
    version of this note said repair is "not mediated by carrier energy"; that was overstated.
  - Across edits that do span the dose range, the data fit a saturating curve plus a non-specific
    offset of about +0.03 to +0.06:

    | edit | carrier overlap | change in AUC (seeds 210 / 211 / 212) |
    |---|---|---|
    | d | about 0.01 | +0.061 / +0.032 / -0.002 |
    | b | 0.25 | +0.026 / +0.078 / +0.002 |
    | c, mean | about 0.62 | +0.117 / +0.105 / +0.119 |
    | a | 1.0 | +0.150 / +0.138 / +0.097 |

  - Removing about 60% of the carrier energy repairs 80-100% as much as removing all of it.
  - **Leakage is ruled out** (checker's recomputation):
    - The carrier is a sinusoid with an AR1 envelope and a small phase random walk.
    - 97.5% of its energy falls in its own 2 DFT bins over patch-aligned 16-sample windows, so the
      "unused" bins hold about 1%.
    - The 116-column left pad is a pure phase shift, so the bins stay orthogonal.
    - rho computed against the ideal bin subspace is the right target.
- **G0b-specific passes by 0.0003,** far inside the AUC sampling error (SE about 0.03 at
  n = 210). Treat it as inconclusive.
- **Consequence.** H-M1-BLM on the patch kernel is uninterpretable in this configuration.
  - The reason: with 4 carriers in about 13 in-band dims, a band-matched null cannot vary carrier
    dose, and partial removal saturates. It does not show that the mechanism is absent.
  - Per the plan, the control gate failing means Stage 1 does not move to real data.
  - Next-step options: docs/decision_log.md item 19.
