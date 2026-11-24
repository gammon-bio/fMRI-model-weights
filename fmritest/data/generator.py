"""Synthetic resting-state fMRI generator with planted networks and a planted site confound.

Generative model for subject s at site c (all quantities are known ground truth):

    z_k(t)      ~ AR(1) with coefficient phi, unit stationary variance       (latent neural sources)
    s_k(t)      = (z_k * h)(t), rescaled to unit variance                    (HRF-convolved sources)
    L_s         = M (elementwise) exp(sigma_load * eps),  eps ~ N(0, 1)      (subject loadings; zeros stay zero)
    signal_s    = S_s L_s^T                                                  (T x R)
    noise_s     = sigma_s * N(0, 1),  sigma_s ~ U(noise_min, noise_max)      (subject noise level)
    confound_s  = strength * a_s(t) u_c^T                                    (site pattern u_c, site time course a_s)
    X_s         = signal_s + confound_s + noise_s

M (R x K) is the network membership matrix. Each region has one primary network and, with
probability `overlap_prob`, a secondary network (overlapping membership).

Site confound modes:
  - "spatial":  site-specific spatial pattern u_c; time course is broadband AR(1), same law at every site.
  - "spectral": spatial pattern shared by all sites; time course is narrowband at a site-specific frequency.
  - "both":     site-specific pattern AND site-specific frequency.
With strength == 0 the confound term is exactly zero and nothing else depends on site.

Planted diagnosis (shortcut task). Each subject gets a binary label y. Patients (y = 1) have the
loadings of one network scaled by (1 + effect) (effect < 0 = hypoconnectivity). The label is
spuriously correlated with site in the training sites (per-site prevalence `train_prevalence`,
exact counts) and decorrelated in the held-out site (`heldout_prevalence`). A classifier that
uses site as a proxy for y therefore looks good on training sites and fails on the held-out site.
Note: with diagnosis enabled the site <-> label correlation is a deliberate site dependence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import gamma


def canonical_hrf(tr: float, duration: float = 32.0) -> np.ndarray:
    """SPM-style canonical double-gamma HRF sampled at TR, normalised to unit sum."""
    t = np.arange(0.0, duration, tr)
    h = gamma.pdf(t, 6.0) - gamma.pdf(t, 16.0) / 6.0
    return h / h.sum()


def ar1(rng: np.random.Generator, n: int, phi: float, burn: int = 100) -> np.ndarray:
    """Stationary unit-variance AR(1) series of length n."""
    e = rng.standard_normal(n + burn) * np.sqrt(1.0 - phi**2)
    x = np.empty(n + burn)
    x[0] = rng.standard_normal()
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def make_membership(rng: np.random.Generator, n_regions: int, n_networks: int, overlap_prob: float,
                    primary_range=(0.6, 1.0), secondary_range=(0.3, 0.6)) -> tuple[np.ndarray, np.ndarray]:
    """Return membership M (R x K) and the primary-network label of each region."""
    primary = np.repeat(np.arange(n_networks), int(np.ceil(n_regions / n_networks)))[:n_regions]
    primary = rng.permutation(primary)
    M = np.zeros((n_regions, n_networks))
    M[np.arange(n_regions), primary] = rng.uniform(*primary_range, size=n_regions)
    for r in range(n_regions):
        if rng.random() < overlap_prob:
            other = rng.choice([k for k in range(n_networks) if k != primary[r]])
            M[r, other] = rng.uniform(*secondary_range)
    return M, primary


@dataclass
class SyntheticDataset:
    X: np.ndarray                 # (N, T, R) observed BOLD, raw (not z-scored)
    signal: np.ndarray            # (N, T, R) network signal only
    confound: np.ndarray          # (N, T, R) site confound only
    noise: np.ndarray             # (N, T, R)
    sources: np.ndarray           # (N, T, K) HRF-convolved latent sources
    diagnosis: np.ndarray         # (N,) planted binary label (all 0 when diagnosis disabled)
    group: np.ndarray             # (N,) 0 = main cohort; 1 / 2 = extra balanced-diagnosis test / val subjects
    membership: np.ndarray        # (R, K) ground-truth membership
    primary_network: np.ndarray   # (R,)
    loadings: np.ndarray          # (N, R, K) subject-specific loadings
    site: np.ndarray              # (N,) site label
    subject_id: np.ndarray        # (N,)
    noise_sd: np.ndarray          # (N,)
    site_patterns: np.ndarray     # (S, R)
    site_freqs: np.ndarray        # (S,) Hz (nan when mode == "spatial")
    config: dict = field(default_factory=dict)


def _confound_timecourse(rng, T, tr, mode, freq, phi):
    if mode == "spatial":
        a = ar1(rng, T, phi)
    else:
        t = np.arange(T) * tr
        # narrowband: carrier at the site frequency with a slowly varying envelope and phase jitter
        env = 1.0 + 0.3 * ar1(rng, T, 0.95)
        phase = rng.uniform(0, 2 * np.pi) + 0.2 * np.cumsum(rng.standard_normal(T)) / np.sqrt(T)
        a = env * np.sin(2 * np.pi * freq * t + phase)
    a = a - a.mean()
    return a / a.std()


def assign_diagnosis(rng, site: np.ndarray, n_sites: int, heldout_site: int, dcfg: dict) -> np.ndarray:
    """Exact-count labels: training sites (in ascending order) get train_prevalence[i] patients."""
    y = np.zeros(len(site), dtype=int)
    train_sites = [c for c in range(n_sites) if c != heldout_site]
    prev = {c: p for c, p in zip(train_sites, dcfg["train_prevalence"])}
    prev[heldout_site] = dcfg["heldout_prevalence"]
    for c in range(n_sites):
        idx = np.flatnonzero(site == c)
        n_pat = int(round(prev[c] * len(idx)))
        y[rng.permutation(idx)[:n_pat]] = 1
    return y


def generate(cfg: dict, seed: int, heldout_site: int | None = None) -> SyntheticDataset:
    """Generate a dataset from a config dict (see configs/default.yaml, key `data`).

    `heldout_site` is needed only for the planted diagnosis (to decorrelate label and site there)."""
    rng = np.random.default_rng(seed)
    R, K, S = cfg["n_regions"], cfg["n_networks"], cfg["n_sites"]
    T, tr = cfg["n_timepoints"], cfg["tr"]
    n_per_site = cfg["subjects_per_site"]
    conf = cfg["confound"]
    strength, mode = float(conf["strength"]), conf["mode"]
    assert mode in ("spatial", "spectral", "both")

    M, primary = make_membership(rng, R, K, cfg["overlap_prob"])
    hrf = canonical_hrf(tr)

    # site parameters (drawn even when strength == 0 so the RNG stream, and hence everything else,
    # is identical between confound-on and confound-off datasets with the same seed)
    pats = rng.standard_normal((S, R))
    if mode == "spectral":
        pats[:] = pats[0]
    pats = pats / np.sqrt((pats**2).mean(axis=1, keepdims=True))
    lo, hi = conf["freq_range"]
    freqs = np.linspace(lo, hi, S)[rng.permutation(S)]
    if conf.get("freq_choices") is not None:
        # opt-in (BrainLM Stage 0b, C1): draw the S site frequencies without replacement from the
        # candidates with a SEPARATE stream; the main-stream permutation above is still consumed, so
        # every other draw is unchanged and the key-absent output stays bit-identical to frozen-v2
        choices = np.asarray(conf["freq_choices"], dtype=float)
        if len(choices) < S:
            raise ValueError("freq_choices needs at least n_sites candidates")
        freqs = np.random.default_rng([seed, 7919]).choice(choices, size=S, replace=False)
    if mode == "spatial":
        freqs = np.full(S, np.nan)

    dcfg = cfg.get("diagnosis", {"enabled": False})
    N = S * n_per_site
    site = np.repeat(np.arange(S), n_per_site)
    # 0 = main cohort; 1 = extra balanced-diagnosis TEST subjects; 2 = extra balanced-diagnosis
    # VALIDATION subjects (for calibration / model selection; never used for reported test metrics)
    group = np.zeros(N, dtype=int)
    on = dcfg.get("enabled", False)
    for g, key in ((1, "n_balanced_per_site"), (2, "n_val_balanced_per_site")):
        n_extra = int(dcfg.get(key, 0)) if on else 0
        if n_extra:
            extra_sites = np.repeat([c for c in range(S) if c != heldout_site], n_extra)
            site = np.concatenate([site, extra_sites])
            group = np.concatenate([group, np.full(len(extra_sites), g)])
    N = len(site)
    # labels come from a separate stream so enabling/disabling the diagnosis never shifts other draws
    drng = np.random.default_rng([seed, 2])
    if dcfg.get("enabled", False):
        if heldout_site is None:
            raise ValueError("diagnosis enabled: pass heldout_site")
        y = np.zeros(N, dtype=int)
        y[group == 0] = assign_diagnosis(drng, site[group == 0], S, heldout_site, dcfg)
        for g in (1, 2):
            for c in np.unique(site[group == g]):
                idx = drng.permutation(np.flatnonzero((group == g) & (site == c)))
                y[idx[: len(idx) // 2]] = 1
    else:
        y = np.zeros(N, dtype=int)
    out = {k: np.zeros((N, T, R)) for k in ("X", "signal", "confound", "noise")}
    sources = np.zeros((N, T, K))
    loadings = np.zeros((N, R, K))
    noise_sd = np.zeros(N)
    pad = len(hrf) + 20
    for n in range(N):
        z = np.stack([ar1(rng, T + pad, cfg["phi"]) for _ in range(K)], axis=1)
        s = np.stack([np.convolve(z[:, k], hrf)[pad:pad + T] for k in range(K)], axis=1)
        s = (s - s.mean(0)) / s.std(0)
        L = M * np.exp(cfg["loading_sd"] * rng.standard_normal((R, K)))
        if y[n]:
            L[:, dcfg["network"]] *= 1.0 + dcfg["effect"]
        sig = s @ L.T
        sd = rng.uniform(*cfg["noise_sd_range"])
        eps = sd * rng.standard_normal((T, R))
        a = _confound_timecourse(rng, T, tr, mode, freqs[site[n]], cfg["phi"])
        amp = rng.uniform(*conf["subject_amp_range"])
        cf = strength * amp * np.outer(a, pats[site[n]])
        out["signal"][n], out["noise"][n], out["confound"][n] = sig, eps, cf
        out["X"][n] = sig + cf + eps
        sources[n], loadings[n], noise_sd[n] = s, L, sd

    return SyntheticDataset(X=out["X"], signal=out["signal"], confound=out["confound"], noise=out["noise"],
                            sources=sources, diagnosis=y, group=group, membership=M, primary_network=primary, loadings=loadings,
                            site=site, subject_id=np.arange(N), noise_sd=noise_sd, site_patterns=pats,
                            site_freqs=freqs, config=dict(cfg, seed=seed))


def zscore_subjects(X: np.ndarray) -> np.ndarray:
    """Per-subject, per-region z-scoring over time (standard preprocessing; uses no cross-subject stats)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def generative_covariance(M: np.ndarray, noise_var: float) -> np.ndarray:
    """Population covariance implied by the confound-free generative model (ignoring loading jitter)."""
    return M @ M.T + noise_var * np.eye(M.shape[0])
