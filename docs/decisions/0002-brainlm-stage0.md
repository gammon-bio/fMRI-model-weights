# ADR 0002: BrainLM Stage 0 implementation choices (2026-11-09)

Status: implemented, unreviewed by Caleb. Conservative defaults taken where the Stage 0 brief or the
BrainLM tutorial was silent or ambiguous. See `docs/brainlm_stage0.md` for results.

- **S1 bf16 autocast on CPU for all heavy forwards and the head fine-tune.** In fp32 the 10-layer
  frozen forward with 730 tokens costs 0.67 s/sample on 2 threads; with bf16 autocast (AMX on this
  Xeon) 0.26 s/sample in the probe, 0.18 s/sample in the run. fp32 for 3 seeds x 28 edited forwards
  would have exceeded the 4 h Stage 0 cap (estimate about 6 h). Relative error of the layer-10 input
  vs fp32: 0.8 % (pooled real-data token mean 0.5 %). The cache path and the edited-kernel path use
  the same numerics, so every delta-AUC is like-for-like. Check on seed 200: fp32 unedited 0.707 vs
  bf16 0.705; fp32 delta-AUC of projection (a) +0.207 vs bf16 +0.212. `--fp32` disables it.
- **S2 Robust scaling precedence.** `toolkit/BrainLM_Toolkit.py` writes `(x - median / IQR)`; we use
  the evidently intended `(x - median) / IQR`, after per-recording temporal demeaning, with per-parcel
  median / IQR pooled over training recordings and time. The tutorial's further division by
  `max_val_to_scale = 5.6430855` is applied.
- **S3 No parcel reordering.** The tutorial sorts A424 parcels by y-coordinate. Synthetic parcels
  have no coordinates and network membership is randomly permuted over regions, so the order is
  left as generated. Stage 1 must apply the coordinate sort.
- **S4 Masking bypassed, padding kept.** Tokens are embedded directly (identical to `ViTMAEModel`
  with mask_ratio 0 and noise = arange; max abs difference 0.0 measured). All 729 patches including
  the -1 padding go through the encoder, as in BrainLM's `modeling_vit_mae_with_padding`, which pads
  and encodes the padded patches too. The head pools the 351 patches that contain any real data
  (tutorial token-range formula); CLS is not used.
- **S5 Early stopping on `val` loss** (in-distribution, label-site correlation intact), weighted by
  the site x label balance weights for the reweighted variant. The balanced sets are never used for
  stopping. Max 8 epochs, patience 2, AdamW, lr 1e-3 head / 5e-5 layers, wd 0.01, batch 16. These
  were fixed before the first run and not tuned.
- **S6 Off-carrier frequencies.** The brief's examples (0.27, 0.40, 0.53, 0.13 Hz) were used as given.
  They sit 0.063-0.07 Hz from the nearest carrier, below the 0.087 Hz resolution of a 16-sample
  window, so they are not fully "off": each 2-dim subspace still carries some carrier energy and all
  four together capture 34 % of the carrier subspace. Reported as measured (`carrier_overlap`).
- **S7 Random-subspace null is destructive, not matched.** A random 8-of-16-dim temporal projection
  removes half of the kernel's temporal content and lowers test_balanced AUC well below unedited. The
  gate compares (a) with that null as specified; the doc also reports the closer comparison with the
  all-four off-carrier projection (same dimension, 34 % carrier overlap).
- **S8 `heavy-check.sh` absent.** `~/.claude/scripts/heavy-check.sh` does not exist on this cloud
  container (4 cores). Load was checked with `uptime` before each launch, and only one heavy job ran
  at a time, at 2 threads.

## Addendum: Stage 0b implementation choices (2026-11-09)

Plan: `docs/brainlm_stage0b_plan.md` (6bfab91). Choices the plan left open:

- **S9 `freq_choices` is set outside `apply_overrides`.** `apply_overrides` refuses keys that are
  absent from `configs/default.yaml`, by design. `default.yaml` was left unchanged, so the opt-in
  key is written into the confound dict after the overrides (`scripts/brainlm_stage0b.py::make_cfg`).
  The generator reads it with `conf.get("freq_choices")`. With the key absent, the output hashes
  (X, diagnosis, site, group, site_freqs) for seeds 0, 1 and 200 are identical before and after the
  edit.
- **S10 Group-balanced reference.** Each epoch draws len(train) indices: a site x label cell
  uniformly, then a member uniformly, with replacement. The loss is unweighted. The Stage 0
  importance-weight path still exists (pass weights, no `group_ids`).
- **S11 Fixed epochs.** `finetune.early_stopping: false` runs all 10 epochs and keeps the final
  weights. In this mode val loss is computed only after the last epoch, for reporting; it plays no
  role in selection. Stage 0 behaviour (`early_stopping` absent = True) is unchanged.
- **S12 Band basis.** The span of the 16-sample DFT bins k = 2..8: cos and sin for k = 2..7, cos only
  at the Nyquist bin k = 8. That gives 13 dims, QR-orthonormal.
  - Edit c: Haar-random 8-dim subspaces of that span, using rng [seed, 99].
  - Edit e: the whole span.
  - Edit d: sin/cos of the 2 candidate bins not drawn for the seed.
- **S13 Carrier share rho** = ||dW Qc||^2 / ||dW||^2 along the kernel's temporal axis, with
  dW = W_edit - W0. This is the definition in `scripts/phase2_confirm_analyze.py::carrier_share`.
- **S14 In-distribution delta.** Each edited forward scores test_balanced and test together in one
  pass (210 + 90 scans).

## Addendum: Stage 0c implementation choices (2026-11-10)

Plan: `docs/brainlm_stage0c_plan.md` (d37c86b + Amendment 1 9eb5217). Code:
`fmritest/brainlm/subspace.py` and `scripts/brainlm_stage0c.py`. Config:
`configs/brainlm_stage0c.yaml`, with threads as a config value; data and fine-tune settings are read
from `configs/brainlm_stage0b.yaml`, unchanged.

- **S15 Paired data.**
  - Each seed is generated twice, the second time with `data.confound.strength` = 0.
  - The script stops if any of these checks fails:
    - signal, noise, diagnosis, site, group, site_freqs and site_patterns are bit-identical;
    - the off-version confound is all zero;
    - X_on equals (signal + confound) + noise exactly.
  - X_on - (X_off + confound) is about 1e-15 because the additions happen in a different order.
  - One robust scaler, fit on the confound-on train subjects, is applied to both versions.
  - The off version is cached only for train and test_balanced.
- **S16 Reader inputs (R2).**
  - The 4 layer readers are the outputs of `layernorm_before` (read by query, key, value) and
    `layernorm_after` (read by `intermediate.dense`) of layers 10 and 11. Each is averaged over the 351
    data tokens per subject.
  - The head reader is the pooled final-LN output, i.e. R1.
  - Sequence: L10 attention, L10 MLP, L11 attention, L11 MLP, head. Each reader's S is estimated with
    every earlier reader already folded.
  - Each reader has its own k.
  - Bias terms are unchanged: the projection acts on the reader's input, W <- W (I - P).
- **S17 Dose nulls (A1.2).** For H1, W_alpha is built from S1. For H2, each reader's W_alpha is built
  from its fixed S2, independently per reader, with the same alpha. The overlap is checked to within
  1e-6 at construction (an assertion). For H2, the reported `overlap` is the mean over the 5 readers.
  The random stream is `default_rng([seed, 4242])`.
- **S18 L** (accepted by the coordinator):
  - LEACE (`fmritest/baselines/leace.py`) is fit on R1 of the unedited vanilla model over the
    train + val pool, with site labels (3 training sites).
  - It is folded as w' = w - w proj, b' = b + w proj mu.
  - Its effective input is r(R1), so D_eff = D (I - proj)^T.
- **S19 m (A1.1).**
  - m is gated on the effective head input:
    - (I - P_W) R1 for H1 and its nulls;
    - r(R1) for L;
    - for H2: R1 after the layer 10-11 edits, then (I - P_head);
    - R1 for K-a and K-e.
  - `m_literal` (R1 activations only) is stored for every edit and is not gated.
  - D0 is R1 of the unedited model on test_balanced (on - off), and S1 is held fixed.
- **S20 Scoring path.**
  - Edit AUCs are computed from the captured R1 (float64) times the head weights. Model-level AUCs use
    `predict`, which runs the head in bf16. The two can differ slightly; every delta is taken within
    the R1 path.
  - K-a and K-e are full forwards. Their reference is the unedited full forward through the same
    path, not the cache.
- **S21 Pooling** (accepted by the coordinator). G0c-metric pools all 85 edits of a seed: 41 H1 +
  41 H2 + L + K-a + K-e. delta(alpha = 0) is the mean over its 10 nulls.
- **S22 Network readout (A1.3).** Not applicable to BrainLM. Off-target damage is reported as the
  change in in-distribution test AUC only.
