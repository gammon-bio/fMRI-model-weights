# BrainLM Stage 0c: plan and gates (fixed before any seed is run)

**Date:** 2026-11-10. Approved by Caleb: "Let's go with D" (docs/open_questions.md item 19).

**Why.** In Stage 0b the patch-kernel mechanism check could not be interpreted: 4 carriers fill
about 13 in-band dims, so a band-matched null cannot vary the carrier dose (docs/brainlm_stage0b.md).
Option D keeps the Stage 0b planting, which passed the shortcut gate, and moves the mechanism check
to the representations that the fine-tuned layers and the head read. There the ground-truth site
subspace comes from paired confound-on/off data.

> Not a scientific result. Synthetic plumbing for the Stage 1 design.

## 1. Fixed settings (unchanged from Stage 0b)

- `configs/brainlm_stage0b.yaml`: freq_choices on 16-sample DFT bins, prevalence [0.8, 0.2, 0.8],
  150 subjects per site, fixed 10 epochs, group-balanced reference.
- Effect -0.3, chosen in Stage 0b on tuning seeds 300 and 301. There is no new tuning.
- Fine-tune of layers 10-11 + LN + head, bf16, 2 threads.
- **Seeds:** 220, 221 and 222. They are fresh.

## 2. Ground-truth site subspace from paired data

**Paired data.**
- For each seed, generate the same dataset twice: confound on, and confound off
  (`data.confound.strength` = 0).
- **Check first:** with the same seed, the signal, noise, labels and site arrays must be
  bit-identical between the two versions. Only the confound term may differ. If this fails, stop
  and report.

**Site difference.** For representation r of subject i, D_i = r_on,i - r_off,i, computed with the
same (unedited vanilla) model on both versions.

**Subspace S_r.**
- The top-k principal directions of {D_i} over **train subjects only**, uncentred.
- k is the smallest number reaching >= 90% of the energy of D, capped at 8.
- Report k and the energy curve.

**Two representation levels:**
- **R1, head input.** The pooled, final-LN output (768-dim) that `head.weight` reads.
- **R2, reader inputs.** The LN-normalised inputs read by every reader downstream of layer 9:
  layer 10 (attention qkv, MLP fc1), layer 11 (attention qkv, MLP fc1) and the head.
  - Token-level D, using the mean over the 351 data tokens per subject for the PCA.
  - Estimated in sequence: layer 11's subspace is estimated with layer 10 already edited.

## 3. Scripted edits (on the vanilla model, all exact weight edits)

| id | edit | ground truth needed |
|---|---|---|
| H1 | `head.weight <- head.weight (I - P_S1)` | yes |
| H2 | every R2 reader W <- W (I - P_S2,reader), i.e. the projection applied after its LN | yes |
| L | LEACE on R1, fit with **site labels** from the agent-visible training pool, folded into `head.weight` / `head.bias` | site labels only (the arm B-static comparator) |
| K-a | Stage 0b patch-kernel projection of all 4 carriers | yes |
| K-e | Stage 0b generic patch-kernel low-pass (k >= 2) | no |

**Dose-response nulls** (run for H1 and H2 separately):
- Subspaces of the same dimension k, W_alpha = orth(cos(theta) Q_S + sin(theta) Q_R).
  - Q_R is a random orthonormal basis orthogonal to S.
  - theta is set so the subspace overlap tr(P_W P_S)/k equals alpha.
- alpha in {0, 0.25, 0.5, 0.75, 1.0}, with 10 random Q_R per alpha (alpha = 1 is H1/H2 itself, run
  once). That gives 41 edits per level.

**Activation-based mechanism metric, defined for any edit:**
- m(edit) = 1 - ||P_S1 D^edited||^2 / ||P_S1 D^unedited||^2.
- D is measured at R1 on **test_balanced** subjects, with S1 kept fixed from the unedited model.
- It is computed for every edit above, including K-a and K-e. It is the candidate Stage 1 mechanism
  measure for agent edits at any location.

**For every edit, report:** change in test_balanced AUC, change in in-distribution test AUC,
change in the network readout (off-target), and m.

## 4. Gates (fixed now)

| gate | pass criterion |
|---|---|
| **G0c-shortcut** | same as G0b-shortcut, on seeds 220-222: mean headroom >= 0.15, mean vanilla gap >= 0.10, reweighted in-distribution test AUC >= 0.70 in every seed |
| **G0c-control** (per level, H1 and H2) | mean [delta(alpha = 1) - mean delta(alpha = 0)] >= 0.05, AND delta(alpha = 1) > max delta(alpha = 0) in >= 2/3 seeds, AND delta(alpha = 1) > 0 in >= 2/3 seeds |
| **G0c-dose** (per level) | within-seed Spearman(alpha, delta AUC) over the 41 edits >= 0.5 in >= 2/3 seeds |
| **G0c-metric** (diagnostic) | pooled over all edits of a seed, Spearman(m, delta AUC) >= 0.5 in >= 2/3 seeds |
| **G0c-compute** | measured seconds per seed |

**Interpretation, fixed now:**
- **G0c-shortcut, and G0c-control plus G0c-dose at at least one level, pass.** Stage 1 can move
  to real data with that level as the mechanism lever:
  - H-M1-BLM is redefined on m or on the per-level overlap;
  - L becomes the arm B-static comparator;
  - K-e becomes the generic-filter comparator.
- **Both levels fail control or dose.** The ground-truth site subspace is not a usable repair lever
  in this model. Stage 1 proceeds only as an outcome-only study (option C), and that needs a
  separate approval.
- **G0c-metric** is reported. It decides whether m can classify agent edits across locations; it
  does not stop the next step.
- **Comparisons of H1/H2 with L and with K-e** are reported, not gated.

## 5. Budget and boundaries

- Wall-time cap 3 h, a single heavy job, threads capped at 2 (a config value). After the first
  seed, the build extrapolates the total and stops if it would exceed 3 h.
- No weights, fine-tuned heads or caches on disk or in git.
- Outputs go to `results/brainlm_stage0c/` and the report to `docs/brainlm_stage0c.md`. Earlier
  results are untouched.
- No changes after any seed has been run. A gate that fails is reported as failed.

## Amendment 1 (2026-11-10, before any seed was run; no outcome seen)

The builder raised two ambiguities before writing the run script. The only data seen so far is the
paired-data check on seed 220, which passed. No AUC has been computed for any edit.

**A1.1. The metric m measures what the head effectively reads.**
- Edits folded into the head (H1, the H1 nulls, L and the head step of H2) do not change the R1
  activations. Read literally, they would score m = 0.
- **Gated definition (G0c-metric):** D is measured on the head's *effective* input.
  - For H1 and its nulls: (I - P_W) R1.
  - For L: the LEACE-transformed R1.
  - For H2: R1 after the layer 10-11 edits, then projected by its head-step subspace.
  - For upstream edits (K-a, K-e, layer edits), effective input = R1.
- The literal R1-only m is stored for every edit as well, reported but not gated.

**A1.2. Dose nulls for H2.**
- Each reader's W_alpha is built from the S2 subspaces fixed by the H2 sequence. They are not
  re-estimated along the null-edited sequence.
- Both choices give H2 at alpha = 1.

**A1.3. Network readout.** The Phase 2 readout (masked region token) has no counterpart in
BrainLM's patch tokens. Off-target damage is reported as the change in in-distribution test AUC
only. This was pre-authorised in the build brief.
