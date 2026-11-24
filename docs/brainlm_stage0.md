# BrainLM Stage 0: synthetic plumbing with the real vitmae_111M weights (2026-11-09)

**Not a scientific result.** Synthetic 424-parcel data only. Purpose: check that the planned Stage 1
setup (docs/brainlm_study_design.md sections 2, 3, 7) works end to end with the official weights,
and measure CPU compute.

**Verdict.**
- **G0-shortcut: FAIL**, after the full iteration budget (two attempts).
- **G0-control: PASS**, on the attempt-0 vanilla models.
- **G0-compute: measured.** About 1960 s per seed for prepare + 28 edited-kernel forwards (CPU, 2 threads, bf16).

## 1. What was built

| file | content |
|---|---|
| `fmritest/brainlm/pack.py` | robust scaling and packing to [B, 3, 432, 432]; 27 x 27 mask of data-carrying patches |
| `fmritest/brainlm/model.py` | encoder loading (sha256 check, mask_ratio 0, no shuffle); layer-9 cache (fp16); `HeadModel`; `train_head` (vanilla / reweighted); `edited_kernel` |
| `fmritest/brainlm/carriers.py` | sin/cos carrier basis on the kernel's temporal axis (16 samples, TR 0.72 s, QR); `project_out`; random bases; subspace overlap |
| `scripts/brainlm_stage0.py` | per-seed pipeline; timings; gates; `--no-control`, `--merge`, `--fp32`, `--fp32-check` |
| `tests/test_brainlm_carriers.py` | 5 weight-free tests: packing and mask, robust scaler, orthonormal bases, projector (idempotent, removes the in-band sinusoid, keeps the orthogonal one), correct-axis check on a synthetic kernel |
| `docs/decisions/0002-brainlm-stage0.md` | implementation choices S1-S8 |
| `results/brainlm_stage0/` | `per_seed.json` (attempt 0, full), `per_seed_attempt1_ep20.json`, `per_seed_attempt2_ep20_eff025.json`, `summary.json` |

`pyproject.toml` gains the optional extra `brainlm = ["transformers==4.46.3"]`. It is installed in
`.venv` only, with `uv pip install --python .venv/bin/python`. `.gitignore` now excludes `*.bin`,
`*.safetensors` and cache directories.

No weights, fine-tuned heads or activation caches were written to disk. The layer-9 cache
(700 x 730 x 768 fp16, 785 MB) lived in RAM only.

**Packing** follows `toolkit/BrainLM_Tutorial.ipynb`:
- `preprocess_images`, notebook lines about 684-708: the input is a [424 parcels, 200 time] matrix,
  divided by 5.6430855 and repeated over 3 channels.
- "Save CLS Tokens" cell, lines about 740-770: `F.pad` with value -1 to 432 x 432. That is 4 rows
  above and below, and 116 columns left and right.
- Token-range cell, lines about 818-832: the data-carrying patches are all 27 rows x columns 7-19,
  i.e. 351 tokens.

So time runs along the image width, and therefore along the **last** kernel axis of
`projection.weight` [768, 3, 16, 16]. The unit test confirms this on a synthetic kernel. The
robust-scaling formula and the missing parcel reordering are explained in ADR S2-S3.

**Model check.**
- 0 missing and 0 unexpected keys; sha256 verified.
- Our no-mask forward equals `ViTMAEModel.forward(mask_ratio=0, noise=arange)` with max abs
  difference 0.0.
- The cache path and the full edited-kernel path with the unedited kernel give the same
  test_balanced AUC: 0.705 / 0.666 / 0.902 by both paths.

**Data.**
- `fmritest.data.generator` via `apply_overrides` with these overrides: n_regions 424, tr 0.72,
  n_timepoints 200, freq_range [0.2, 0.6].
- Unchanged frozen-v2 settings: 100 subjects/site, effect -0.4, strength 0.6, 70 test_balanced and
  30 val_balanced per training site.
- At 424 regions the generator behaves sensibly:
  - networks of 85/85/85/85/84 parcels, 90 parcels with a secondary network;
  - site patterns with unit RMS;
  - the confound spectrum peaks at the assigned carrier, e.g. 0.333 Hz for site 1 at seed 200;
  - 1.3 % of network-signal power lies above 0.15 Hz.
- No config fix was needed. The generator is unchanged.

**Splits.** `make_splits`, held-out site = seed % 4. Sizes: train 180, val 60, test 60,
test_site 100, test_balanced 210, val_balanced 90.

**Fine-tune.**
- Copies of layers 10-11 + final layernorm + a linear head on the mean of the 351 data tokens.
- AdamW: lr 1e-3 (head), 5e-5 (layers), weight decay 0.01, batch 16, patience 2.
- Early stopping on `val` loss, weighted with site x label balance weights for the reweighted
  variant.
- bf16 autocast (ADR S1).

## 2. Results, attempt 0 (fixed settings, 8 epochs): the only run with the positive control

AUC; v = vanilla, rw = reweighted.

| seed (held-out, low-prev site) | test_bal v / rw | test v / rw | test_site v / rw | val_bal v / rw | headroom (test_bal) |
|---|---|---|---|---|---|
| 200 (0, 2) | 0.705 / 0.871 | 0.916 / 0.923 | 0.727 / 0.907 | 0.691 / 0.863 | **+0.166** |
| 201 (1, 2) | 0.666 / 0.739 | 0.929 / 0.820 | 0.673 / 0.643 | 0.658 / 0.673 | +0.074 |
| 202 (2, 1) | 0.902 / 0.698 | 0.985 / 0.514 | 0.976 / 0.938 | 0.922 / 0.746 | -0.204 |

**Site decodability** (multinomial logistic, train -> test, chance 0.33):
- head input of vanilla: 1.00 / 1.00 / 0.98;
- head input of reweighted: 1.00 / 1.00 / 0.98;
- frozen layer-9 token mean: 1.00 in all seeds.

The probe is saturated at strength 0.6, so it carries no information about which model "uses"
site.

**Positive control on the vanilla model.** Each row is a full forward of test_balanced through the
edited model. Delta AUC is relative to the unedited full forward. "Overlap" is the fraction of the
8-dim carrier subspace that the projected subspace captures.

| edit (dims, overlap) | seed 200 | seed 201 | seed 202 |
|---|---|---|---|
| unedited AUC | 0.705 | 0.666 | 0.902 |
| (a) all 4 site carriers (8, 1.00) | **+0.212** (-> 0.917) | **+0.229** (-> 0.895) | **+0.047** (-> 0.949) |
| (b) low-prevalence site carrier only (2, 0.25) | -0.040 | -0.023 | -0.074 |
| (c) 20 random 8-dim subspaces: mean (sd), max | -0.130 (0.042), -0.015 | -0.055 (0.055), +0.097 | -0.157 (0.056), -0.038 |
| (a) percentile among the 20 random subspaces | 100 | 100 | 100 |
| (d) off 0.27 / 0.40 / 0.53 / 0.13 Hz (2 each, 0.03-0.07) | -0.053 / -0.005 / -0.049 / -0.067 | -0.023 / -0.016 / -0.032 / +0.003 | +0.011 / -0.002 / -0.029 / -0.032 |
| (d) all four off frequencies (8, 0.34) | +0.030 | +0.043 | -0.029 |
| in-distribution test AUC, unedited -> (a) | 0.916 -> 0.904 | 0.929 -> 0.951 | 0.985 -> 0.940 |

**fp32 check** (seed 200): unedited 0.707, (a) 0.914, delta +0.207. The bf16 delta was +0.212.

**Reading.**
- Projection (a) brings vanilla test_balanced AUC to 0.90-0.95 in all three seeds. That is above the
  reweighted fine-tune in every seed, and it costs little in-distribution AUC.
- Random 8-dim projections are destructive (mean -0.06 to -0.16). They remove half of the kernel's
  temporal content. So the gate margin of (a) over the random mean (+0.20 to +0.34) partly reflects
  damage done by the null, not only repair by (a).
- The closer, same-dimension comparison is (d)-all-four, which has 34 % carrier overlap. (a) beats
  it by +0.18 / +0.19 / +0.08.
- Removing only the low-prevalence site's carrier (b) does not repair, and hurts slightly in all
  seeds. The shortcut is not carried by that one carrier alone.
- The brief's off-carrier frequencies are within 0.07 Hz of a carrier, below the 0.087 Hz
  resolution of a 16-sample window (ADR S6).

## 3. Iteration on G0-shortcut (settings chosen by val_balanced headroom only)

Rule: at most two settings, at most two attempts. Test sets were not used to choose settings.

| attempt | change | seeds | headroom test_bal | headroom val_bal | vanilla test - test_bal | verdict |
|---|---|---|---|---|---|---|
| 0 | none (8 epochs) | 200-202 | +0.166 / +0.074 / -0.204, mean **+0.012** | +0.172 / +0.015 / -0.176 | +0.21 / +0.26 / +0.08 | FAIL |
| 1 | epochs 8 -> 20 | 200, then stopped | -0.041 | -0.052 | +0.04 | FAIL |
| 2 | epochs 20 + effect -0.4 -> -0.25 | 200-202 | +0.145 / +0.055 / +0.052, mean **+0.084** | +0.179 / +0.017 / +0.110, mean +0.102 | +0.32 / +0.30 / +0.27 | FAIL |

**Why each change was made.**
- **Attempt 0.** Both fine-tunes were still improving at the epoch cap (best epoch 7-8 of 8). The
  reweighted model was unstable: seed 202 reached in-distribution AUC 0.514.
- **Attempt 1.** With 20 epochs, the vanilla model on seed 200 learned the planted network effect
  itself: test_balanced 0.950, val_balanced 0.959. At effect -0.4 the true signal is easy for
  BrainLM features, so the shortcut fades with training. Val_balanced headroom was -0.052, so this
  attempt was stopped after seed 200 to save budget.
- **Attempt 2.** The weaker effect (-0.25) restored a clear shortcut in vanilla: test 0.82-0.89 vs
  test_balanced 0.54-0.57. But the reweighted reference stayed weak:
  - test_balanced 0.60-0.72;
  - early stops at epochs 3-9;
  - seed 202 in-distribution AUC 0.369, consistent with over-weighting the minority cells (6 per training site in train, 2 in val; recounted in review) of the
    low-prevalence site.
  Mean headroom was 0.084 < 0.15.

Iteration budget exhausted. Stopped as instructed. The positive control was not re-run on the
attempt-2 models: a full control run takes about 20 more minutes per seed, which would have pushed
Stage 0 past about 4 h.

**Diagnosis.** The planted shortcut does form: in every attempt-0 and attempt-2 seed, vanilla
in-distribution AUC is clearly above its test_balanced AUC. What fails is the **reweighted
reference**, which is noisy with 180 training subjects and a 90/10/90 prevalence (minority cells
of 6 subjects per training site in train and 2 in val; corrected in review, was "2"). The scripted carrier projection beats it in every attempt-0 seed. For
Stage 1 this argues for:
- more training subjects per site, or a milder prevalence (e.g. 0.8/0.2/0.8);
- a group-DRO or upsampling ceiling instead of importance weights;
- a fixed epoch budget, not val early stopping, for the ceiling.
These are suggestions only. None was tried here: each would be a third setting.

## 4. Gates

| gate | criterion | measured | result |
|---|---|---|---|
| G0-shortcut | mean headroom >= 0.15 AND vanilla test clearly above test_balanced | attempt 0: headroom +0.012, gap +0.19. Attempt 2: headroom +0.084, gap +0.30 | **FAIL** (headroom part; the gap part holds) |
| G0-control | mean delta(a) - random mean >= 0.05 AND (a) beats all 20 random in >= 2/3 seeds | +0.276 (+0.342 / +0.284 / +0.203); beats all 20 in 3/3 seeds | **PASS** (attempt-0 models; see the caveat on the destructive null) |
| G0-compute | measured s/seed for prepare + nulls | 1752 / 1873 / 2246 s, mean 1957 s (attempt 0) | **measured** |

## 5. Compute (measured, CPU 2 threads, Intel Xeon 2.1 GHz with AMX, bf16 autocast)

| stage | per seed (attempt 0: seeds 200 / 201 / 202) |
|---|---|
| encoder load + sha256 | 2.4 s (once) |
| generate (700 subjects) | 2.4 / 2.7 / 4.5 s |
| robust scale | 1.3 s; packing 0.03 s per 32 subjects |
| cache layer-9 (700 scans) | 128 / 127 / 116 s (0.17-0.18 s/scan) |
| fine-tune vanilla (8 epochs) | 250 / 281 / 352 s (about 32 s/epoch + val) |
| fine-tune reweighted (8 epochs) | 243 / 292 / 253 s |
| head evals (4 sets x 2 models) | 37 / 40 / 39 s total |
| site probes | 57 / 56 / 54 s |
| one edited-kernel forward of test_balanced (210 scans) | 40.3 / 41.7 / 54.7 s mean |
| 28 edited forwards (unedited + a + b + 20 random + 5 off) | 1128 / 1169 / 1520 s |
| **prepare + controls** | **1752 / 1873 / 2246 s** |
| fp32 check (2 forwards, seed 200) | 303 s |
| seed wall | 2173 / 1992 / 2384 s |

In attempt 2 (20 epochs, early stopping) prepare took 462-700 s. In attempt 1 (20 epochs, no
early stop for vanilla) it took 1300 s. Without bf16, the 10-layer forward is 2.6x slower: 0.67 vs
0.26 s/scan in the probe.

**Total Stage 0 wall time.** 05:11 to 07:59, about 2.8 h, including attempts 1 and 2. Exactly one
heavy job ran at a time.

**Extrapolation for a 30-seed study.** These are formula estimates from the measurements above,
not measured runs.
- Per seed = prepare P + (N nulls + about 8 comparators) x 45.6 s. The 45.6 s is the mean
  edited-forward time on 210 scans.
- N = 50:
  - P = 685 s: 3330 s/seed (0.93 h), so **28 h** for 30 seeds;
  - P = 1300 s: 33 h.
- N = 200:
  - P = 685 s: 10,170 s/seed (2.8 h), so **85 h**;
  - P = 1300 s: 90 h.
- Edits confined to layers 10-11 or the head need only the cached head forward (about 8 s per 210
  scans). Nulls on those tensors therefore cost about 5x less than patch-kernel nulls.
- At these rates, the patch-kernel nulls of a 30-seed study need a GPU (design §7), or N = 50 run
  over about 1-1.5 days of single-job CPU time.

## 6. Deviations and open issues

- bf16 autocast throughout (ADR S1). Validated against fp32 on seed 200 only.
- Off-carrier frequencies are not well separated from the carriers at 16-sample resolution
  (ADR S6). The random null is destructive (ADR S7).
- The positive control was measured only on attempt-0 vanilla models. Attempt-0 vanilla is partly
  undertrained (best epoch 7-8 of 8).
- The harness (`fmritest/harness`) was not wired to BrainLM in this stage. "The harness scores
  BrainLM edits" (design §2) remains open.
- `~/.claude/scripts/heavy-check.sh` does not exist on this container. Load was checked with
  `uptime` instead (ADR S8).

## 7. Review (main session + independent Fable 5.1 check, 2026-11-09)

**Gate arithmetic: confirmed.** The checker recomputed both gates from `per_seed.json`: headroom
0.166 / 0.074 / -0.204, mean 0.012; control margin 0.342 / 0.284 / 0.203, mean 0.276, and 3/3 seeds
beating all 20 random subspaces. Two caveats:
- "Vanilla test clearly above test_balanced" was put into code as a mean gap >= 0.10. The brief
  never fixed that number, and seed 202 alone (gap 0.083) would fail it.
- Attempt 2 changed the data (effect -0.25), not a fine-tune setting. The script logs
  test_balanced per seed while it runs, so "settings chosen on val_balanced only" is the builder's
  statement and cannot be checked independently.

**The positive control does not show ground-truth repair.**
- The generator draws site frequencies as `linspace(0.2, 0.6, 4)[permutation]`
  (`fmritest/data/generator.py:136`). So the carrier set {0.2, 0.333, 0.467, 0.6} Hz is the same
  in every seed, and projection (a) is the same 8-dim subspace in every seed (overlap 0.9999999).
- 98% of (a) lies in the span of the 16-sample DFT bins >= 0.174 Hz.
- So (a) needs no seed-specific knowledge. An agent could hard-code it.
- It is **not**, however, a generic low-pass. (a) is 8 of the 13 band dims, and the data show
  specificity at 8 dims:
  - 20 random 8-dim subspaces overlap (a) by 0.48-0.52, yet give -0.055 to -0.157;
  - the same-band off-carrier projection (8 dims, 34% overlap) gives +0.030 / +0.043 / -0.029,
    against +0.212 / +0.229 / +0.047 for (a).
- A full 13-dim low-pass was never run. Whether it would repair as well is open, and plausible,
  since only 1.3% of network power lies above 0.15 Hz.

**The G0-shortcut failure has three causes, not one.**
1. **Reweighted overcorrection.** In seed 202 the reweighted in-distribution AUC was 0.514 in
   attempt 0 and 0.369 in attempt 2.
2. **Vanilla did not use the shortcut in seed 202** (gap 0.083), and the shortcut is transient:
   at 20 epochs on seed 200, vanilla learned the true signal.
3. **At effect -0.25 the true signal may not be learnable** from 180 subjects, which would also
   cap the reweighted model. Vanilla test_balanced AUC was 0.54-0.57.

Weighted early stopping on val cells of 2 subjects is the noisiest part of the procedure.

**Consequences for Stage 1** are recorded as proposals in `docs/brainlm_study_design.md` §10. They
are not locked; see `docs/decision_log.md` item 18.
