# BrainLM weight-space repair study: design (DRAFT, not pre-registered)

**Status:** D1-D6 decided 2026-11-09 (the recommended options; see docs/decision_log.md item 17). Stage 0 done, G0-shortcut FAIL (docs/brainlm_stage0.md); proposed changes in §10. Nothing here is locked. Once the decisions in §9 are made,
a pre-registration in the style of `docs/phase2_confirm_prereg.md` follows.

**Question.** Can an LLM agent find and repair a known shortcut inside a real fMRI foundation model,
BrainLM vitmae_111M, by editing its weights? And does the synthetic-model result carry over? That
result was that repair succeeds when the agent removes the planted scanner carrier at the input
layer (`results/phase2_confirm/REPORT.md`).

**Why ground truth first.** The synthetic studies show the agent always "finds" something: 0/5 said
"neither" in the false-positive arm. On a real model with a real label and no answer key, a gain
cannot be told apart from luck or overfitting. So the confirmatory part plants a flaw with a known
mechanism into real data. The no-answer-key version (real ABIDE diagnosis across sites) comes last
and is exploratory.

## 1. What carries over from the synthetic studies

| synthetic model | BrainLM vitmae_111M |
|---|---|
| region-as-token, `embed.weight` 64 x 48 acts on a 48-TR window | patch embedding `vit.embeddings.patch_embeddings.projection.weight` 768 x 3 x 16 x 16 acts on 16 parcels x 16 timepoints |
| carrier subspace C = sin/cos at 4 site frequencies over 48 samples | carrier subspace over the **16-sample temporal axis** of each patch kernel (all 3 channels and 16 parcel rows) |
| shortcut learned by fine-tuning the whole model + head | shortcut learned by fine-tuning the top K encoder layers + head (§3) |
| mechanism metric rho on `embed.weight` | rho on the patch kernel's temporal axis, plus the site subspace in the fine-tuned layers' residual stream (`scoring.py` style) |

**Constraint from 16-sample patches.**
- At BrainLM's TR of about 0.72 s, frequency resolution is 1/(16 x 0.72), about 0.087 Hz. Carriers
  must be at least about 0.09 Hz apart, and the Nyquist limit is 0.69 Hz.
- H-M3 showed a carrier inside the neural band (0.04 Hz) is hard to separate from signal. Plan:
  - 3 carriers in 0.10-0.45 Hz, outside the dominant BOLD band;
  - a pre-specified 0.04-0.06 Hz carrier as a hard stratum.

## 2. Stages

**Stage 0: plumbing, no real data. Can start now.**
- Input: our generator scaled to 424 parcels at TR 0.72 s, packed into BrainLM's 432x432 frame.
- Checks:
  - the fine-tune (§3) learns the shortcut;
  - the harness scores BrainLM edits;
  - the scripted carrier projection on the patch kernel works as a positive control;
  - per-episode compute on CPU is feasible.
- Not a scientific result: the data are synthetic.

**Stage 1: semi-synthetic on real data. Confirmatory, pre-registered.**
- **Data:** real resting-state scans, A424-parcellated and robust-scaled per the BrainLM tutorial.
  Use controls only (or one diagnostic group) so the real label adds no structure (see §9 D1).
- **Planted label:** "patients" get one real network's parcels (e.g. A424 default-mode parcels)
  scaled by (1 + effect), as in the synthetic generator.
- **Planted site:** a site-specific spatial pattern x narrow-band carrier at a site-specific
  frequency, injected into each scan. Prevalence is correlated with site at the training sites and
  balanced in held-out test sets (`test_balanced`).
- **Ground truth known:** carrier frequencies, spatial patterns, the label mechanism.
- **Real nuisance structure:** real scanner and site differences, motion residue, real spectra.
- **Tasks:** the same three as Phase 2.
  - T-FIX: repair, scored by sealed test_balanced AUC.
  - T-READ: which model uses site.
  - A false-positive arm with the confound off.

**Stage 2: real labels, no answer key. Exploratory only.**
- Task: ABIDE ASD vs TD; train on some sites and score on held-out sites.
- The agent tries to improve held-out-site AUC.
- Comparators: ComBat harmonisation, LEACE-site, reweighted fine-tune.
- Matched random-edit nulls are mandatory. A positive result is reported as "beats nulls and
  harmonisation", never as "found the mechanism".

## 3. Model and fine-tune

**Recommendation:** freeze patch embedding + layers 0..(11-K); fine-tune layers (12-K)..11 + a
linear head on the mean of non-padded tokens; K = 2 to start.

**Why:**
- **CPU feasibility.** Cache layer-(11-K) outputs once per scan window (about 0.35 s per sample for
  the full forward on 2 threads). Then training touches only K layers.
- **A realistic practice:** freeze the backbone, adapt the top.
- **Two repair routes.** The shortcut lives in the fine-tuned layers and head, but the carrier still
  enters through the frozen patch embedding. The agent can either fix the readout (top layers / head)
  or remove the carrier at the input. That is exactly the contrast H-M1 tested.
- **Alternatives:**
  - full fine-tune on a GPU, closer to "the model learned it end to end";
  - LoRA on all layers, with the shortcut in low-rank adapters (a small delta, easy to ship).
  - Decision D3.

**Comparators (as in Phase 2):**
- unedited;
- reweighted fine-tune (the ceiling);
- post-hoc LEACE-site on the head input, fit on the site-correlated pool;
- scripted ground-truth carrier projection on the patch kernel (positive control);
- matched random edits on the same tensors (null).

## 4. Agent arms and delivery

- **Arm A (weights only, synthetic probes).**
  - The Phase 2 primary. Likely much harder at 112M parameters.
  - The agent must run BrainLM to probe it. A numpy forward pass of a 12-layer ViT is feasible but
    slow.
  - Decision D4: allow torch for inference with gradients disabled (audit item (c) becomes
    "autograd / backward / requires_grad used").
- **Arm B-static (agent gets its training pool).**
  - The agent-visible, site-correlated labelled pool is shipped as files. That is what a
    practitioner holding their own training data has, and it needs no live oracle, so it works with
    cloud sessions.
  - The fair comparator is post-hoc LEACE fit on the same pool.
  - Recommended as a co-primary arm for BrainLM.
- **Delivery:**
  - The base weights (450 MB) are too big for a git branch (100 MB file limit).
  - The agent downloads them from Hugging Face and checks sha256
    25f5b52178f7409dee4fed2fa3dd18c6692d0764d96996ff0ee4be740520fcd6.
  - That requires running agent sessions in the `fMRI-model-test` environment (HF allowlisted)
    instead of Default.
  - The episode branch carries only the fine-tuned top-K layers + head (K = 2: about 14M params,
    about 57 MB fp32, about 28 MB fp16), TASK/ARCH docs, and the B-static pool if that arm runs.
  - The audit adds: no network beyond pip and the one HF file.
- **License (CC BY-NC-ND 4.0).**
  - Fine-tuned and edited weights are derivatives: keep them only in this **private** repo and the
    sealed episode store. Never publish them.
  - Results, code and synthetic data can be published.
  - Episode branches must be deleted or kept private after the study.

## 5. Hypotheses to pre-register (sketch, Stage 1)

- **H-A-BLM (arm A and arm B-static separately).** Agent edit vs matched random-edit null on sealed
  test_balanced AUC (Wilcoxon of percentile - 50).
- **H-B-BLM (arm B-static).** Agent edit vs post-hoc LEACE fit on the same pool. Two-sided, with an
  equivalence margin pre-specified (e.g. +/-0.03 AUC).
- **H-M1-BLM.** Edits with carrier share >= 0.5 on the patch kernel's temporal axis beat other edits.
  Ceiling-adjusted sensitivity and covariate balance as in the confirmation prereg.
- **H-R-BLM.** Read task above chance (binomial).
- **Positive control:** scripted carrier projection improves AUC. If it does not, the planted lever is
  not present in this model, and H-M1-BLM is uninterpretable.
- **Power:** 30 included seeds per arm. Effect sizes are unknown at 112M, so the Stage 0/pilot
  numbers set the final n.

## 6. Data (Stage 1/2)

- **Recommended: ABIDE Preprocessed** (fcp-indi). Open access with a registration-free S3 mirror,
  real multi-site structure, and the real ASD label needed for Stage 2.
  - Pipeline: download `func_preproc` (MNI space) for one subject at a time, parcellate with
    `toolkit/atlases/A424+2mm.nii.gz` (resample the atlas to the data grid), keep only the 424 x T
    float32 array (about 0.5 MB), and delete the NIfTI. This fits the about 9 GB disk.
  - Hosts to allowlist: `fcp-indi.s3.amazonaws.com` (and possibly `s3.amazonaws.com`).
  - **TR mismatch:** ABIDE sites use TR 1.5-3 s vs BrainLM's 0.72 s, so the time series must be
    resampled. The planted carrier is injected after resampling, at the model's sampling rate.
  - Data-use terms: non-commercial, with attribution. Cite ABIDE / PCP.
- **Alternatives:**
  - HCP young adult: TR 0.72 s, an exact match. Needs a ConnectomeDB data-use agreement, which Caleb
    must sign; no credentials in sessions.
  - OpenNeuro resting-state datasets: open, varied TR, no diagnosis label.

## 7. Compute and cost (estimates; Stage 0 measures them)

- **Prepare one seed (CPU, 2 threads):**
  - cache layer-9 activations for about 600 scan windows: about 4 min;
  - fine-tune 2 layers + head: tens of minutes, to be measured;
  - comparators + nulls: 200 matched edits x test-set forward through K layers (or 12 layers for
    patch-kernel edits). The patch-kernel nulls need full forward passes, about 210 x 0.35 s x 200
    = about 4 h per seed on CPU. **Reduce:** 50 nulls (already near-inert in Phase 2), batch
    inference, or a GPU.
- **Agents:** larger models mean longer episodes; expect $3-6 per Opus episode (vs $1.85).
  - 30 seeds x 2 arms + read + false-positive: about 75 episodes, roughly $250-450 at Opus prices.
  - This exceeds the remaining cloud credit; Decision D6.
- **A GPU** (Caleb's Mac via MPS, one job at a time per CLAUDE.md, or a cloud GPU) shortens
  preparation from hours to minutes per seed.

## 8. Plan

1. **Stage 0** (now, no decisions needed beyond this doc): synthetic 424-parcel data into BrainLM.
   - Test the K = 2 fine-tune, the shortcut, the scripted patch-kernel positive control, and harness
     scoring.
   - Write `docs/brainlm_stage0.md` with measured compute.
2. Caleb decides D1-D6.
3. Data pipeline for the chosen dataset, with a QC report (spectra, site separability, tSNR).
4. Stage 1 pilot: 2 seeds x arms, harness bugs, cost. Not counted.
5. Pre-register Stage 1 (Fable pre-lock review), lock, run, blind audit with full paging, analyse,
   Fable verification.
6. Stage 2 (exploratory) if Stage 1 shows the agent can repair BrainLM at all.

## 9. Decisions needed from Caleb

- **D1, data source:** ABIDE Preprocessed (recommended; needs `fcp-indi.s3.amazonaws.com`
  allowlisted), HCP (DUA, exact TR), or synthetic-only for now.
- **D2, Stage 1 population:** controls only (cleanest planted label) vs all subjects with the real
  diagnosis regressed out.
- **D3, fine-tune scheme:** top-K (recommended for CPU) vs full fine-tune on GPU vs LoRA.
- **D4, agent tooling:** allow torch inference without autograd (recommended), or numpy-only as in
  Phase 2.
- **D5, arms:** arm A only, or arm A + arm B-static (recommended).
- **D6, budget and compute:** the Stage 1 agent budget (about $250-450 for the full design; a pilot is
  about $20), and whether a GPU is available for preparation.

## 10. Changes proposed after Stage 0 (2026-11-09; not yet adopted)

From `docs/brainlm_stage0.md` §7 (main-session review plus an independent Fable check), ranked:

1. **Randomise and seal carrier frequencies per seed.** Keep them at least 0.087 Hz apart, ideally
   on 16-sample DFT bins (0.26 / 0.347 / 0.434 / 0.521 Hz). A fixed set lets an agent hard-code
   the fix without finding anything.
2. **Make the positive control specific.** It must beat:
   - a same-dimension off-band projection;
   - the generic DFT low-pass (bins >= 0.174 Hz) with no ground truth;
   - a band-matched random null. The current random null is destructive.
3. **H-M1-BLM classification.** With 8-of-16-dim temporal edits, carrier share >= 0.5 is at chance
   level (random overlap 0.48-0.52). Use the percentile against matched random edits, or rho >= 0.8.
4. **Drop the 0.04-0.06 Hz hard stratum** (§1). That is under 0.7 cycles in 16 samples, so it
   cannot be separated from DC and trend.
5. **Fix the headroom regime, then require G-shortcut to pass on the pilot before pre-registering.**
   - A fixed epoch budget for both models.
   - Upsampling or group-DRO instead of importance weights.
   - Milder prevalence (e.g. 0.8/0.2/0.8) and/or more subjects per site.
   - An effect size that leaves the true signal learnable.
   - Re-run Stage 0 on the revised settings (about 1 h CPU for 3 seeds without the full null sweep).
