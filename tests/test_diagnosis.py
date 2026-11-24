import numpy as np
import pytest

from fmritest.config import apply_overrides, load_config
from fmritest.data.generator import generate
from fmritest.data.splits import make_splits
from fmritest.eval.diagnosis import balance_weights, diag_eval


@pytest.fixture
def dcfg(data_cfg):
    data_cfg["subjects_per_site"] = 20
    data_cfg["diagnosis"]["n_balanced_per_site"] = 10
    return data_cfg


def test_diagnosis_labels_and_groups(dcfg):
    h = 2
    d = generate(dcfg, 0, heldout_site=h)
    main = d.group == 0
    tr_sites = [c for c in range(4) if c != h]
    for c, p in zip(tr_sites, dcfg["diagnosis"]["train_prevalence"]):
        assert d.diagnosis[main & (d.site == c)].mean() == pytest.approx(p, abs=0.03)
    assert d.diagnosis[main & (d.site == h)].mean() == 0.5
    extra = d.group == 1
    assert extra.sum() == 30 and h not in set(d.site[extra])
    for c in tr_sites:
        assert d.diagnosis[extra & (d.site == c)].mean() == 0.5


def test_diagnosis_effect_applied_only_to_patients(dcfg):
    d = generate(dcfg, 1, heldout_site=0)
    k, eff = dcfg["diagnosis"]["network"], dcfg["diagnosis"]["effect"]
    dcfg["diagnosis"]["effect"] = 0.0
    d0 = generate(dcfg, 1, heldout_site=0)
    pat = d.diagnosis == 1
    np.testing.assert_allclose(d.loadings[pat][..., k], d0.loadings[pat][..., k] * (1 + eff))
    np.testing.assert_allclose(d.loadings[~pat], d0.loadings[~pat])
    np.testing.assert_array_equal(d.diagnosis, d0.diagnosis)


def test_main_cohort_unchanged_by_extras(dcfg):
    dcfg["diagnosis"]["n_balanced_per_site"] = 0
    a = generate(dcfg, 3, heldout_site=1)
    dcfg["diagnosis"]["n_balanced_per_site"] = 10
    b = generate(dcfg, 3, heldout_site=1)
    np.testing.assert_array_equal(a.X[a.group == 0], b.X[b.group == 0])


def test_splits_route_balanced_extras(dcfg):
    d = generate(dcfg, 0, heldout_site=3)
    sp = make_splits(d.site, 3, seed=0, group=d.group, strata=d.diagnosis)
    assert set(sp["test_balanced"]) == set(np.flatnonzero(d.group == 1))
    for k in ("train", "val", "test", "test_site"):
        assert not np.any(d.group[sp[k]] == 1)
    allids = np.concatenate(list(sp.values()))
    assert len(allids) == len(np.unique(allids)) == len(d.site)


def test_diagnosis_requires_heldout(dcfg):
    with pytest.raises(ValueError):
        generate(dcfg, 0)


def test_balance_weights_decorrelate():
    site = np.array([0] * 10 + [1] * 10)
    y = np.array([1] * 9 + [0] + [1] + [0] * 9)
    w = balance_weights(site, y)
    for c in (0, 1):
        m = site == c
        assert np.isclose((w * y)[m].sum(), (w * (1 - y))[m].sum())


def test_diag_eval_detects_pure_shortcut():
    rng = np.random.default_rng(0)
    n = 400
    site = rng.integers(0, 2, n)
    y = np.where(rng.random(n) < np.where(site == 0, 0.9, 0.1), 1, 0)
    sp = {"train": np.arange(0, 200), "test": np.arange(200, 300)}
    yb = y.copy()
    yb[300:] = rng.integers(0, 2, 100)  # balanced test, independent of site
    sp["test_balanced"] = np.arange(300, 400)
    F = {k: np.c_[site[v] + 0.01 * rng.standard_normal(len(v)), rng.standard_normal(len(v))] for k, v in sp.items()}
    r = diag_eval(F, yb, site, sp, C=1.0, seed=0)
    assert r["test"]["bacc"] > 0.8 and abs(r["test_balanced"]["bacc"] - 0.5) < 0.12


def test_apply_overrides(cfg):
    c = apply_overrides(cfg, {"data.confound.strength": 0.0, "data.overlap_prob": 0.8})
    assert c["data"]["confound"]["strength"] == 0.0 and c["data"]["overlap_prob"] == 0.8
    assert cfg["data"]["confound"]["strength"] == 0.6
    with pytest.raises(KeyError):
        apply_overrides(cfg, {"data.nope": 1})


def test_val_balanced_group(dcfg):
    dcfg["diagnosis"]["n_val_balanced_per_site"] = 8
    d = generate(dcfg, 0, heldout_site=1)
    sp = make_splits(d.site, 1, seed=0, group=d.group, strata=d.diagnosis)
    assert set(sp["val_balanced"]) == set(np.flatnonzero(d.group == 2)) and len(sp["val_balanced"]) == 24
    assert 1 not in set(d.site[d.group == 2])
    for c in (0, 2, 3):
        assert d.diagnosis[(d.group == 2) & (d.site == c)].mean() == 0.5
    assert not set(sp["val_balanced"]) & set(sp["test_balanced"])
