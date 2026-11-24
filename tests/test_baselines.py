import numpy as np
from sklearn.linear_model import LogisticRegression

from fmritest.baselines.fc import components_from_affinity, fc_reconstruct, lagged_cov, lagged_reconstruct, varimax
from fmritest.baselines.leace import LeaceEraser
from fmritest.baselines.probes import site_probe
from fmritest.eval.metrics import network_recovery


def test_fc_reconstruct_exact_for_noiseless_low_rank():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((10, 3))
    S = rng.standard_normal((5000, 3))
    X = S @ A.T
    C = np.cov(X.T)
    x = (rng.standard_normal((4, 3, 20)).transpose(0, 2, 1) @ A.T).transpose(0, 2, 1)  # (4, 10, 20)
    mask = np.zeros((4, 10), bool)
    mask[:, :2] = True
    pred = fc_reconstruct(C, x, mask, ridge=1e-9)
    np.testing.assert_allclose(pred, x, atol=1e-4)


def test_affinity_extraction_recovers_block_structure():
    M = np.kron(np.eye(3), np.ones((5, 1)))
    A = M @ M.T + 0.05 * np.random.default_rng(0).standard_normal((15, 15))
    C = components_from_affinity(A, 3, "eig_varimax")
    assert network_recovery(C, M)["score"] > 0.95
    # attention-like affinity: uniform floor plus block structure, rows sum to 1
    P = 0.02 + 0.1 * M @ M.T
    P = P / P.sum(1, keepdims=True)
    assert network_recovery(components_from_affinity(P, 3), M)["score"] > 0.95


def test_varimax_preserves_communalities():
    L = np.random.default_rng(1).standard_normal((12, 3))
    np.testing.assert_allclose((varimax(L) ** 2).sum(1), (L ** 2).sum(1), rtol=1e-6)


def test_leace_removes_linear_information():
    rng = np.random.default_rng(0)
    z = rng.integers(0, 3, 3000)
    X = rng.standard_normal((3000, 8)) + 3.0 * np.eye(8)[z]
    er = LeaceEraser().fit(X, z)
    Xe = er.transform(X)
    Z = np.eye(3)[z]
    cross = (Xe - Xe.mean(0)).T @ (Z - Z.mean(0)) / len(X)
    assert np.abs(cross).max() < 1e-8
    acc_before = LogisticRegression(max_iter=2000).fit(X, z).score(X, z)
    acc_after = LogisticRegression(max_iter=2000).fit(Xe, z).score(Xe, z)
    assert acc_before > 0.9 and acc_after < 0.4
    # minimal edit: orthogonal-to-concept variation is preserved
    assert np.linalg.matrix_rank(er.proj) == 2


def test_site_probe_null_and_signal():
    rng = np.random.default_rng(0)
    site = np.repeat(np.arange(3), 20)
    subj_tr, subj_te = np.repeat(np.arange(0, 60, 2), 3), np.repeat(np.arange(1, 60, 2), 3)
    noise = rng.standard_normal((60, 5))
    feat = lambda s, sig: (noise[s] + sig * np.eye(5)[site[s]] + 0.1 * rng.standard_normal((len(s), 5)))
    r = site_probe(feat(subj_tr, 3.0), subj_tr, feat(subj_te, 3.0), subj_te, site, n_perm=50)
    assert r["acc"] > 0.9 and r["p_perm"] < 0.05
    r0 = site_probe(feat(subj_tr, 0.0), subj_tr, feat(subj_te, 0.0), subj_te, site, n_perm=50)
    assert r0["p_perm"] > 0.05


def test_lagged_reconstruct_lag0_equals_fc_and_context_helps():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 3))

    def make(n, T):
        S = rng.standard_normal((n, T, 3))
        for t in range(1, T):
            S[:, t] += 0.8 * S[:, t - 1]
        X = S @ A.T + 0.5 * rng.standard_normal((n, T, 8))
        return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)

    Xtr, Xte = make(60, 200), make(6, 48).transpose(0, 2, 1)
    mask = np.zeros((6, 8), bool)
    mask[:, :2] = True
    C = np.cov(Xtr.reshape(-1, 8).T, bias=True)
    np.testing.assert_allclose(lagged_reconstruct(lagged_cov(Xtr, 0), 0, Xte, mask, ridge=1e-6),
                               fc_reconstruct(C, Xte, mask), atol=1e-6)
    e0 = ((lagged_reconstruct(lagged_cov(Xtr, 0), 0, Xte, mask) - Xte) ** 2)[mask].mean()
    e2 = ((lagged_reconstruct(lagged_cov(Xtr, 2), 2, Xte, mask) - Xte) ** 2)[mask].mean()
    assert e2 < e0
