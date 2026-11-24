# NOTES -- pilot-s04, T-FIX arm A (budget 0, weights + ARCH.md only)

## Deliverable
`edit.npz` replaces two tensors: `head.weight` (1, 64) and `head.bias` (1,). No backbone tensor is
touched, so `network_readout_delta` is exactly 0. The edit is a rank-2 projection of the head vector
plus a re-centred bias:

    w_new = w - (w . u1) u1 - (w . u2) u2        with u1, u2 orthonormal (see below)
    b_new = -(w_new . hbar)                       hbar = mean pooled head_in over a synthetic mixture

    |w| = 0.894 -> |w_new| = 0.499,  cos(w_new, w) = 0.559,  w.u1 = -0.668,  w.u2 = +0.322
    b: +0.0783 -> -0.0952
    sha256(edit.npz) = 2e11ea64973dbbd4e0f97cccac72f30557a87eec5a14570dadf442e31ed55995

u1 = the single direction in head_in space along which global temporal smoothing moves the pooled
representation ("spectral / predictability axis"). u2 = the linear-drift direction, orthogonalised
against u1 (keeps the original head's drift invariance, which the u1 projection alone would break).

## What I did (no data, no oracle; all probing is my own numpy forward pass on synthetic inputs)
1. Re-implemented the forward pass from ARCH.md in numpy (LN, 4-head attention, exact-erf GELU,
   4-window subject mean). Checked that the edited model reproduces logit = head_in . w + b and that
   all backbone hooks are unchanged.
2. Inspected weights. `region_emb` has strong low-rank structure; hierarchical clustering gives four
   region groups: {0,2,4,5,6,8,9,12,13,18,23,27}, {11,15,19,22,25}, {3,10,14,21,28}, {1,7,16,17,20,24,26,29}.
   Attention patterns are essentially static (across-subject SD ~0.01), so cross-region information
   reaches the head through fixed mixing plus the MLP nonlinearity.
3. Probed what moves the logit, using a synthetic generator (module latents x loadings + noise, with
   smoothing / AR(1) / drift / global signal / spikes / steps / oscillations as knobs). Findings, in SD
   units of the subject-level logit on a fixed background:
   - global temporal smoothing sigma 1 -> 0.3: +4.7 SD; sigma 1 -> 2.5: -2.0 SD; white-noise level
     x2: +6.4 SD; spikes: +2.8 SD; AR(1) rho 0.8: -0.8 SD.
   - this whole response is ONE direction in head_in (first singular value 11.5 vs 0.98 for the next),
     stable across five connectivity backgrounds (cosine > 0.994), cosine with the head = 0.75.
   - it is region-specific: smoothing only regions {11,15,19,22,25} gives -4.5 SD, smoothing the
     12-region group gives -0.4 SD. Coupling / noise changes confined to that same region set move the
     logit by about 1 SD (gain 1 -> 0.4: +1.3 SD; noise x2: +1.0 SD); other groups: < 0.3 SD.
   - the head is almost exactly invariant to linear drift (cos -0.01), quadratic drift, and
     variance steps although the backbone represents them strongly (|delta head_in| ~ 2). That is a
     learned cancellation, so the training generator very likely contains such nuisances.
   - block-1 MLP (neurons 25, 110, 108, 63, 14, 123, ...) and block-1 attention carry the feature; no
     neuron separates "global smoothness" from "region-set coupling" -- same sign pattern everywhere.
   Interpretation: the model computes "temporal predictability (smoothness / SNR) of one region set"
   and the head reads mostly that. Scanner-like global properties move it 3-6x more than
   network-specific changes do, which is consistent with the stated collapse under site balancing.
4. Candidate edits (all closed-form projections of head.weight; nothing fitted to data):
   erase u1 only; u1 + drift; u1 + drift + extra nuisance directions (variance step, 20-TR
   oscillation, global signal); a joint "spectral" direction pooled over smoothing/AR/noise; and
   erasing the uniform-coupling direction. Evaluated with fresh seeds on simulated 2-site x 2-dx
   designs. Erasing the uniform-coupling direction removes all diagnosis-like signal (AUC 0.5):
   rejected. u1-only unmasks a +3 SD drift response: rejected. Extra directions cost diagnosis
   sensitivity faster than they remove leakage: rejected. u1 + drift kept.
5. Bias: centred so the new logit has mean 0 over a mixture of synthetic subjects (smoothing 0.5-2.5,
   noise 0.7-1.4, drift 0.2-1.5, region-set gain 1.0 or 0.6). This is a blind guess at the sealed
   data's centre; the new logit's SD is only ~0.13, so balanced accuracy at logit > 0 is fragile.

## Held-out check of the exact submitted head (fresh seed, 200 subjects per cell, dx = region-set gain 1 vs 0.5)
    site type   orig balancedAUC / siteAUC / within-site AUC    edited balancedAUC / siteAUC / within-site AUC
    smoothing        0.570 / 0.000 / 0.641                            0.562 / 0.347 / 0.562
    AR(1)            0.592 / 0.001 / 0.683                            0.591 / 0.436 / 0.589
    noise level      0.595 / 0.985 / 0.673                            0.640 / 0.752 / 0.646
The edit removes most of the site response (the logit is no longer site-shifted) but the surviving
diagnosis-like signal is weaker, because in this model the two are carried by the same feature.
Net effect on pooled balanced AUC in simulation: 0.00 to +0.045 for spectral/SNR site effects and
coupling-like diagnoses; -0.02 to -0.04 if the site effect is drift- or global-signal-like or the
diagnosis is itself a global dynamics property.

## Astra's rule, honestly applied
I have no real data, so no gain was measured on data not used to choose the edit. On my own
simulations the gain of the u1 component exceeds its standard error only when the site effect is
spectral/SNR-like and the diagnosis effect is not tiny; the drift component's gain exceeds SE only
when drift is present in the data (which the learned invariance suggests). I kept both on the
mechanistic argument above, not on a measured gain. The registered prediction reflects that:
a small expected increase in test_balanced_auc with a 90% interval that includes no change.

## Limits
- All probing used my own generator; the true generator (what "site" and "diagnosis" are) is unknown.
- Residual site leakage remains (site AUC 0.35-0.75 in simulation) because the head_in response is
  nonlinear and the diagnosis-like and site-like responses share one feature.
- test_auc / test_bacc on the original correlation will drop because the site shortcut is removed.
- Within-site AUC (and hence test_site_auc) is expected to drop slightly.
