import numpy as np
import pytest


@pytest.fixture
def data_cfg(data_cfg):
    data_cfg["diagnosis"]["enabled"] = False  # these tests concern the network/confound generator only
    return data_cfg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fmritest.baselines.probes import fc_features
from fmritest.data.generator import canonical_hrf, generate, zscore_subjects


def test_hrf_shape():
    h = canonical_hrf(2.0)
    t = np.arange(len(h)) * 2.0
    assert 4.0 <= t[h.argmax()] <= 6.0           # peak ~5 s
    assert h.min() < 0 and t[h.argmin()] > 10.0   # late undershoot


def test_shapes_and_determinism(data_cfg):
    data_cfg["subjects_per_site"] = 5
    a, b = generate(data_cfg, 3), generate(data_cfg, 3)
    N = 5 * data_cfg["n_sites"]
    assert a.X.shape == (N, data_cfg["n_timepoints"], data_cfg["n_regions"])
    assert a.membership.shape == (data_cfg["n_regions"], data_cfg["n_networks"])
    np.testing.assert_array_equal(a.X, b.X)
    assert not np.allclose(a.X, generate(data_cfg, 4).X)
    np.testing.assert_allclose(a.X, a.signal + a.confound + a.noise)
    assert (a.membership > 0).sum(1).min() >= 1                    # every region in >= 1 network
    assert ((a.membership > 0).sum(1) > 1).any()                   # overlap present


def test_strength_zero_has_no_confound_and_same_signal(data_cfg):
    data_cfg["subjects_per_site"] = 5
    data_cfg["confound"]["strength"] = 0.0
    off = generate(data_cfg, 1)
    data_cfg["confound"]["strength"] = 0.6
    on = generate(data_cfg, 1)
    assert np.all(off.confound == 0)
    np.testing.assert_array_equal(off.signal, on.signal)   # identical RNG stream
    np.testing.assert_array_equal(off.noise, on.noise)


def test_empirical_fc_matches_generative_structure(data_cfg):
    data_cfg["confound"]["strength"] = 0.0
    data_cfg["subjects_per_site"] = 20
    d = generate(data_cfg, 0)
    X = zscore_subjects(d.X)
    fc = np.mean([np.corrcoef(x.T) for x in X], 0)
    # implied covariance: E[L L^T] + noise, converted to correlation
    L = d.loadings
    cov = np.einsum("nrk,nqk->rq", L, L) / len(L) + np.mean(d.noise_sd ** 2) * np.eye(L.shape[1])
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    iu = np.triu_indices_from(fc, 1)
    assert np.corrcoef(fc[iu], corr[iu])[0, 1] > 0.95
    same = d.primary_network[:, None] == d.primary_network[None, :]
    off = ~same
    np.fill_diagonal(same, False)
    assert fc[same].mean() > fc[off].mean() + 0.2


def _site_cv_acc(d):
    F = fc_features(zscore_subjects(d.X))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    return cross_val_score(clf, F, d.site, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()


@pytest.mark.parametrize("seed", [0, 1])
def test_strength_zero_site_not_decodable_from_fc(data_cfg, seed):
    data_cfg["confound"]["strength"] = 0.0
    data_cfg["subjects_per_site"] = 30
    d = generate(data_cfg, seed)
    acc = _site_cv_acc(d)
    # chance = 0.25 with 120 subjects; binomial 99th percentile ~ 0.35
    assert acc < 0.36, acc


def test_confound_on_site_decodable_from_fc(data_cfg):
    data_cfg["confound"]["strength"] = 0.6
    data_cfg["subjects_per_site"] = 30
    assert _site_cv_acc(generate(data_cfg, 0)) > 0.8


@pytest.mark.parametrize("mode", ["spatial", "spectral"])
def test_confound_modes(data_cfg, mode):
    data_cfg["confound"]["mode"] = mode
    data_cfg["subjects_per_site"] = 3
    d = generate(data_cfg, 0)
    if mode == "spectral":
        assert np.allclose(d.site_patterns, d.site_patterns[0])
        # dominant confound frequency matches the site frequency
        a = d.confound[0] @ d.site_patterns[0]
        f = np.fft.rfftfreq(len(a), d.config["tr"])
        assert abs(f[np.abs(np.fft.rfft(a)).argmax()] - d.site_freqs[d.site[0]]) < 0.01
    else:
        assert np.all(np.isnan(d.site_freqs))
        assert not np.allclose(d.site_patterns[0], d.site_patterns[1])


def test_freq_choices_opt_in(data_cfg):
    """C1 (BrainLM Stage 0b): with freq_choices the S site frequencies are 4 distinct candidates that
    vary across seeds; everything else in the dataset is unchanged by the option."""
    import copy
    data_cfg["subjects_per_site"] = 3
    cand = [k / (16 * 0.72) for k in range(2, 8)]
    cfg_c = copy.deepcopy(data_cfg)
    cfg_c["confound"]["freq_choices"] = cand
    sets = []
    for seed in range(6):
        ds = generate(cfg_c, seed)
        f = ds.site_freqs
        assert len(f) == data_cfg["n_sites"] == 4
        assert len(np.unique(f)) == 4 and np.all(np.isin(f, cand))
        sets.append(tuple(f))
        base = generate(data_cfg, seed)
        assert np.array_equal(base.site_patterns, ds.site_patterns)
        assert np.array_equal(base.signal, ds.signal) and np.array_equal(base.noise, ds.noise)
    assert len(set(sets)) > 1                                   # differs across seeds
    assert len({frozenset(s) for s in sets}) > 1                # not just a reordering of one set
