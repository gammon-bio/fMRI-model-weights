import numpy as np
import pytest

from fmritest.data.splits import check_no_leakage, make_splits


@pytest.mark.parametrize("seed", range(5))
def test_splits_disjoint_and_site_heldout(seed):
    site = np.repeat(np.arange(4), 50)
    h = seed % 4
    sp = make_splits(site, h, seed=seed)
    allids = np.concatenate(list(sp.values()))
    assert len(allids) == len(np.unique(allids)) == 200
    assert set(site[sp["test_site"]]) == {h}
    for k in ("train", "val", "test"):
        assert h not in set(site[sp[k]])
        assert len(set(site[sp[k]])) == 3          # all training sites represented


def test_leakage_detected():
    site = np.repeat(np.arange(4), 10)
    sp = make_splits(site, 0)
    bad = dict(sp, test=np.concatenate([sp["test"], sp["train"][:1]]))
    with pytest.raises(AssertionError):
        check_no_leakage(bad, site, 0)
    bad = dict(sp, train=np.concatenate([sp["train"], sp["test_site"][:1]]))
    with pytest.raises(AssertionError):
        check_no_leakage(bad, site, 0)
