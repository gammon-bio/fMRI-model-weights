# Changelog

## 2026-11-24 (latest) -- review fixes for the public README and export script (not yet exported)
- An independent review of public commit 1ce1fec recomputed every headline number (all matched)
  and verified all lock hashes. It raised wording issues. Fixed in README.md:
  - the Fable stratum claim: only the no-change difference sits in the 0.04 Hz stratum;
  - the main-study lock covers analysis rules, not code; code was first locked for the
    confirmation; the stale "(DRAFT)" header and the 41abc1a/48f0102 commits are explained;
  - the sign-test caveat on H-A;
  - H-M1 is described as an association, and the overall gain did not clearly replicate;
  - labels for exploratory results, what a reader can verify, the Fable audit method, the
    deferred arms, and the main report's pre-re-audit audit statement.
- scripts/export_public.sh now reads its marker list from an untracked, gitignored
  .export_markers file and fails if that file is missing.
- No numbers, locked files, results files or analyses were changed.

## 2026-11-24 -- public-export branch (claude/public-export)
- Preparing a public snapshot as a new repo, fMRI-model-weights. The private history stays here.
- Removed from this branch: CLAUDE.md and .claude/ (private working notes). docs/open_questions.md
  is renamed to docs/decision_log.md, with all references updated.
- Added:
  - LICENSE (MIT; BrainLM weights and ABIDE data are excluded);
  - a canary GUID;
  - a rewritten README covering every study, plus figures from scripts/make_readme_figures.py;
  - scripts/export_public.sh, which builds a single-commit snapshot and refuses if any private
    file or marker is present.
- Fixed test_task_md_has_no_root_and_root_is_inferred. It now checks the isolation label against
  configs/harness.yaml instead of the pre-amendment string. No locked file was changed.

## 2026-11-10 -- Fable 5.2 agent pilot (exploratory)
- Plan docs/fable52_pilot_plan.md (28c806c) committed before any episode.
- 15 prepared, 10 included (fab-fix-*). All served by claude-fable-5-2; blind audit 10/10 VALID.
  $26.24 total.
- Locked confirmation analysis vs Opus confirmation:
  - ON rate 4/10 vs 7/27 (p 0.44);
  - ΔAUC mean +0.024 vs +0.029 (p 0.74);
  - H-M1 inconclusive (OFF n = 0).
- Post hoc, threshold-fragile: Fable either edits the mechanism (4/4 ON) or makes a deliberate
  no-op under Astra's rule (6/10). The difference sits entirely in the 0.04 Hz stratum.
- No evidence that Fable repairs better. Report: docs/fable52_pilot.md. Fable-checked.

## 2026-11-10 -- BrainLM Stage 0c review
- Gates recomputed independently, and by a Fable check: exact match. Locked rule: Stage 1 may use H2.
- Review in docs/brainlm_stage0c.md §8:
  - H2 changes sign by seed; the reference is not a ceiling.
  - The dose pass is carried by the nulls; the full edit is a discontinuity.
  - m ranks the five real edits negatively in every seed (-0.40 / -0.50 / -0.10). Not usable as is.
  - Off-target damage is not gated.
- Open question 20: four required changes before the Stage 1 lock.

## 2026-11-10 -- BrainLM Stage 0c (plan d37c86b + Amendment 1 9eb5217; synthetic plumbing)
- Paired confound-on/off check passes in all seeds (bit-identical except for the confound term).
- New `scripts/brainlm_stage0c.py` and `configs/brainlm_stage0c.yaml`. They cover S1 (head input) and
  S2 (every reader downstream of layer 9, estimated in sequence), the edits H1 / H2 / L / K-a / K-e,
  and 41-edit dose nulls per level. m is measured on the effective head input (gated); the literal
  m is stored but not gated.
- Seeds 220-222, effect -0.3:
  - G0c-shortcut PASS (headroom 0.172, gap 0.213).
  - H1: control FAIL (-0.073) and dose FAIL (1/3).
  - H2: control PASS (+0.131; 2/3 seeds) and dose PASS (Spearman 0.66 / 0.97 / 0.86).
  - G0c-metric PASS 2/3 (0.53 / 0.24 / 0.88).
  - Compute about 2420 s/seed; total 2 h 03 min.
- Caveats:
  - H2 hurts in seed 220 (-0.108).
  - The alpha = 0 null is nearly inert.
  - Full removal at R1 costs about 0.27 of in-distribution AUC.
  - m saturates at 1 for both kernel edits.
- Report: docs/brainlm_stage0c.md. ADR 0002 addendum S15-S22.

## 2026-11-09 -- BrainLM Stage 0b review
- Fable check: gates correct, no selection leakage; carrier 97.5% in-bin, so leakage is ruled out.
  Softened the review claim to: the band-matched null cannot vary carrier dose, and the data fit a
  saturating curve plus a non-specific offset. Stage 1 options D/A/C in decision-log item 19.

## 2026-11-09 -- BrainLM Stage 0b (plan 6bfab91; synthetic plumbing)
- C1: opt-in `data.confound.freq_choices` in the generator (separate RNG [seed, 7919]). With the key
  absent, output hashes for seeds 0 / 1 / 200 are identical to before (= generator-frozen-v2), and
  there is a new unit test. C2/C3: group-balanced sampling and fixed epochs in `fmritest/brainlm/model.py`
  (the Stage 0 paths are kept). Config: `configs/brainlm_stage0b.yaml`; script: `scripts/brainlm_stage0b.py`.
- C5 tuning (seeds 300 / 301): mean val_balanced headroom -0.4: 0.139, -0.3: 0.200, so effect -0.3 was
  chosen by the rule.
- Gate seeds 210-212:
  - **G0b-shortcut PASS** (headroom 0.189, vanilla gap 0.182, reference test >= 0.82).
  - **G0b-control FAIL** (d(a) - mean d(c) = +0.015; d(a) > max d(c) in 1/3 seeds). Random 8-dim
    subspaces inside the 13-dim band repair almost as much as the true carriers.
  - **G0b-specific PASS** by 0.0003 (0.0303).
  - Compute: about 2490 s/seed. In-band random rho is 0.48-0.73 (chance level).
- Report: docs/brainlm_stage0b.md. ADR 0002 addendum S9-S14.

## 2026-11-09 (latest) -- BrainLM Stage 0 review
- Independent Fable check: gate arithmetic confirmed. The positive control uses a seed-invariant
  carrier set (generator linspace + permutation), so it does not show ground-truth repair. It does
  show 8-dim specificity within the band. The shortcut FAIL has 3 causes. Minority cells corrected
  to 6 train / 2 val. Stage 1 proposals in design §10; decision-log item 18.

## 2026-11-09 -- BrainLM Stage 0 (synthetic plumbing, real vitmae_111M weights)
- New `fmritest/brainlm/` (pack, model, carriers), `scripts/brainlm_stage0.py`,
  `tests/test_brainlm_carriers.py`, ADR 0002. Optional extra `brainlm` (transformers 4.46.3) in .venv.
- Seeds 200-202, 424 parcels, TR 0.72, carriers 0.2-0.6 Hz. bf16 autocast on CPU (validated vs fp32
  on seed 200).
- G0-shortcut FAIL after the full iteration budget (attempt 0 mean headroom +0.012; attempt 2, 20
  epochs + effect -0.25, +0.084). The shortcut forms; the reweighted reference is the weak part.
- G0-control PASS: projecting all 4 carriers out of the patch kernel lifts vanilla test_balanced AUC
  by +0.21 / +0.23 / +0.05 (to 0.90-0.95), above all 20 random subspaces in 3/3 seeds. The random
  null is destructive; vs the same-dimension off-carrier projection the margin is +0.08 to +0.19.
- G0-compute: about 1960 s/seed for prepare + 28 edited forwards. Estimated 30 seeds x 50 nulls
  about 28 h CPU, x 200 nulls about 85 h. Report: docs/brainlm_stage0.md.

## 2026-11-09 -- Main-study re-audit; BrainLM design draft
- **Full-coverage blind re-audit of the 55 main-study transcripts:** 54 VALID; main-fix-001 VOID for
  `git branch -a`, which the original auditor had cleared.
  - H-A sensitivity without it: p = 0.0051 (originally 0.0032).
  - Report: results/phase2_main/reaudit/README.md.
  - The auditors were run as a 6-agent workflow; Caleb had approved the re-audit, but not the use of
    the workflow tool.
- **docs/brainlm_study_design.md** (draft):
  - Stage 0: synthetic plumbing.
  - Stage 1: semi-synthetic planted shortcut on real A424 data.
  - Stage 2: exploratory real ABIDE labels.
  - Decisions D1-D6 are pending.

## 2026-11-09 -- Phase 2 confirmation study (pre-registered) and BrainLM 111M
- docs/phase2_confirm_prereg.md: Fable pre-lock review, fixes applied, locked (48f0102) before any
  seed was prepared.
- Episodes: 41 prepared, 30 included, 30 Opus cloud sessions. The blind audit used full transcript
  paging: 27 VALID, 3 VOID (`git branch -a`).
- **H-M1 supported:**
  - ON +0.121 vs OFF -0.005, p = 0.0036;
  - ceiling-adjusted p = 0.014;
  - stratified p = 0.031.
- **H-M2:** p = 0.016. **H-M3:** p = 0.003. The positive control passed.
- **H-A replication:** p = 0.009, but the overall gain was only +0.029. Only 7/27 agents found the
  carrier edit.
- A Fable checker reproduced H-M1 and the class assignments exactly.
- Report: results/phase2_confirm/REPORT.md.
- **Audit-tool gap:** list_events returns the newest page only. Fixed for this study. The main study's
  audit coverage is not established for long transcripts (decision-log item 16).
- BrainLM vitmae_111M downloaded, hash-verified and loaded on CPU: stock ViTMAE, 112.6M params. See
  docs/brainlm_feasibility.md.
- New personal cloud environment `fMRI-model-test` with the Hugging Face hosts allowlisted. Agent
  sessions still run in Default.

## 2026-11-04 -- Input-layer check (exploratory) and BrainLM feasibility scout
- `scripts/phase2_input_layer_check.py`, results in results/phase2_main/input_layer_check/REPORT.md.
  Post-unblinding and not pre-registered.
  - **Scripted fix:** a ground-truth carrier projection on embed.weight gives +0.155 AUC (CI
    0.125-0.179, 28/30 seeds). Random subspaces give -0.007; off-carrier frequencies give -0.001.
  - **Agent embed edits:** 98% of edit energy lies in the carrier subspace (chance 17%). No
    seed-selection effect was found.
  - **Failure mode:** the low-prevalence carrier at 0.04 Hz.
  - **Transplant to other seeds:** +0.057.
  - Run 1 had a capture-keying bug; it is archived as `input_layer_check_run1_buggy_capture/`.
- docs/brainlm_feasibility.md:
  - huggingface.co is blocked by this environment's network policy, so the 111M/650M weights were not
    downloaded.
  - The repo's 5M dev checkpoint loads and runs on CPU (about 2.5 s per sample) with BrainLM's custom
    code and transformers 4.46.3.
  - Hooks work.
  - License: CC BY-NC-ND 4.0.

## 2026-11-04 -- Phase 2 main study (Opus 5.5, arm A)
- Episodes: 55 Opus episodes on per-episode orphan branches; all audit-valid (blind audit before
  scoring).
- **H-A:** p = 0.0032, mean AUC gain +0.070.
- **H-R:** 19/20, p = 2e-5.
- **False-positive arm:** 0/5 answered "neither".
- Fable verification reproduced all numbers.
- Corrections:
  - "matches LEACE" -> "not distinguishable";
  - the percentile-100 reading is caveated (the null is near-inert);
  - the stale isolation label in meta.json is documented, and the config is fixed for future runs.
- Report: results/phase2_main/REPORT.md.

## 2026-11-03 -- generator-frozen-v2
- Decisions (docs/decision_log.md): proceed despite tightened-G1 FAIL; report G1 both ways.
- Power calibration on validation data only (seeds 100-103, `val_balanced`): effect size alone cannot
  reach headroom >= 0.25; frozen 100 subjects/site, effect -0.4, 70 balanced test (n = 210) + 30
  balanced val subjects per training site. Validation headroom 0.280 (3/4 seeds >= 0.25).
- Test result (5 seeds): per-episode SE max 0.0345 (target met); headroom 0.195 +/- 0.144 (target
  NOT met; 2/5 seeds >= 0.25). Not retuned on test data.
- `hard` dropped: one-pass recalibration on all 4 sites closest 0.794 (target 0.75 +/- 0.03).
- Gate: G1 original PASS, G1 tightened FAIL, G2-G4 PASS, negative control OK.
- Open questions moved to docs/decision_log.md and tracked.

## 2026-11-03 (latest) -- diagnosis fine-tuning (Phase 2 repair target)
- `fmritest/finetune.py` (DiagnosisModel subclasses the transformer; hook names unchanged),
  `scripts/finetune.py`, `configs/finetune.yaml`, tests.
- confound_on, 5 seeds: vanilla fine-tune 0.88 in-distribution vs 0.51 correlation-broken (pure site
  shortcut). Reweighted fine-tune +0.09 (5/5 seeds); post-hoc LEACE-site 0.59; within-site signal 0.60.
  Headroom is real but small relative to per-episode noise (SE ~0.065).

## 2026-11-03 (later) -- v2: diagnosis shortcut, hard condition, stricter gates
- v1 sweep (10 runs) archived to `results/v1/`: gate GO, but an independent (Fable 5.1) review showed
  G2 is passed by untrained models (masked-query readout inherits input FC) and G1's edge over pooled
  FC is mostly site-specific covariance + temporal context. Also fixed: remove_direction(head=) on
  non-hook_z hooks; in-model LEACE now fit on masked + unmasked tokens.
- New baselines: lagged (+/-2 TR) and per-site conditional-Gaussian reconstruction. G1 now vs the
  best of these. G2 now also requires beating 10 random-init models.
- Planted diagnosis (network-0 loadings x0.7 for patients; prevalence 0.9/0.1/0.9 at training sites,
  0.5 at held-out site and in 20 extra balanced subjects per training site). New G4.
- `hard` condition (overlap 0.8, noise SD 1.0-1.6, loading SD 0.4). Miscalibrated: pilot used two
  sites only; real FC recovery 0.886, not ~0.75.
- v2 gate: **NO-GO** (G1 fails: model - best linear MSE +0.007 +/- 0.005, 0/5 seeds). G2, G3, G4 pass.
- Frozen final-layer embeddings carry ~no diagnosis signal (within-site CV 0.50-0.54 vs FC 0.70-0.74),
  so the diagnosis repair task has no headroom in its current form. See OPEN_QUESTIONS.md.
- `make reproduce JOBS=2` re-run: 15/15 runs identical to the reported sweep except timing fields.

## 2026-11-03 -- Phase 1 build + Phase 2 protocol
- Synthetic generator (`fmritest/data`): 30 regions, K=5 overlapping networks, AR(1) sources
  convolved with a double-gamma HRF, TR=2 s, 200 TRs; per-subject loadings/noise; S=4 sites with a
  spatial+spectral site confound (strength configurable, 0 = none). Ground truth returned.
- Model (`fmritest/models`): 2-layer, d=64, 4-head region-as-token masked reconstruction
  transformer with named hook points.
- Baselines (`fmritest/baselines`): conditional-Gaussian FC reconstruction; FC eig+varimax,
  PCA and temporal ICA network recovery; linear site probes with permutation null; LEACE.
- Eval (`fmritest/eval`), interp API (`fmritest/interp`), scripted gate (`scripts/gate.py`),
  figures, `make reproduce`.
- Forking paths disclosed in README: training steps raised 3000 -> 10000 after a seed-0 smoke
  run (val loss not converged); affinity-extraction bug (eigenvalue clipping, then a
  double-centring step that discards one network dimension) found and removed before the
  gated 5-seed run.
- `docs/protocol.md`: Phase 2 agent protocol (not run).
