import numpy as np

from fmritest.eval.metrics import mean_sd, network_recovery, recovery_with_null


def test_hungarian_handles_permutation_and_sign():
    rng = np.random.default_rng(0)
    M = rng.random((30, 5))
    C = -M[:, [3, 0, 4, 1, 2]]
    r = network_recovery(C, M)
    assert abs(r["score"] - 1.0) < 1e-12
    assert r["assignment"] == [1, 3, 4, 0, 2]


def test_null_below_true_and_pvalue():
    rng = np.random.default_rng(0)
    M = np.kron(np.eye(5), np.ones((6, 1)))
    r = recovery_with_null(M + 0.1 * rng.standard_normal(M.shape), M, 200, 0)
    assert r["score"] > 0.9 and r["null_q95"] < 0.6 and r["p_perm"] < 0.01
    rnd = recovery_with_null(rng.standard_normal(M.shape), M, 200, 0)
    assert rnd["p_perm"] > 0.01


def test_mean_sd():
    s = mean_sd([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and abs(s["sd"] - 1.0) < 1e-12
