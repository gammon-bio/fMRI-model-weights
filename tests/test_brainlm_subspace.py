"""BrainLM Stage 0c subspace helpers (weight-free, synthetic)."""
import numpy as np
import pytest
import torch

from fmritest.baselines.leace import LeaceEraser
from fmritest.brainlm import subspace as S


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_overlap_construction_hits_alpha(alpha):
    rng = np.random.default_rng(0)
    Q_S, _ = np.linalg.qr(rng.standard_normal((768, 5)))
    for _ in range(5):
        W = S.overlap_basis(rng, Q_S, alpha)
        np.testing.assert_allclose(W.T @ W, np.eye(5), atol=1e-10)
        assert abs(S.overlap(W, Q_S) - alpha) < 1e-6
        P_W, P_S = S.projector(W), S.projector(Q_S)
        assert abs(np.trace(P_W @ P_S) / 5 - alpha) < 1e-6


def test_energy_basis_k_and_cap():
    rng = np.random.default_rng(1)
    U, _ = np.linalg.qr(rng.standard_normal((50, 50)))
    s = np.array([10, 5, 2] + [0.1] * 47)
    D = rng.standard_normal((200, 50))
    D = (np.linalg.qr(D)[0] * s) @ U.T
    _, k, curve = S.energy_basis(D)
    assert k == 2 and curve[1] >= 0.9 > curve[0]       # 100+25 of 129.47 = 0.965
    _, k, _ = S.energy_basis(rng.standard_normal((300, 50)))
    assert k == 8                                       # isotropic: capped


def test_fold_projection_after_layernorm_equals_projecting_ln_output():
    torch.manual_seed(0)
    ln = torch.nn.LayerNorm(32)
    ln.weight.data.uniform_(0.5, 1.5)
    ln.bias.data.normal_()
    lin = torch.nn.Linear(32, 12)
    Q, _ = np.linalg.qr(np.random.default_rng(2).standard_normal((32, 3)))
    x = torch.randn(7, 9, 32)
    y = ln(x)
    ref = lin(y - y @ torch.from_numpy(Q @ Q.T).float())
    lin2 = torch.nn.Linear(32, 12)
    lin2.load_state_dict(lin.state_dict())
    lin2.weight.data = torch.from_numpy(S.fold_projection(lin.weight.data.double().numpy(), Q)).float()
    torch.testing.assert_close(lin2(ln(x)), ref, atol=1e-5, rtol=1e-5)


def test_leace_folding_equals_leace_on_activations():
    rng = np.random.default_rng(3)
    z = rng.integers(0, 3, 400)
    X = rng.standard_normal((400, 20)) + np.eye(3)[z] @ rng.standard_normal((3, 20)) * 2
    er = LeaceEraser().fit(X, z)
    w, b = rng.standard_normal((1, 20)), rng.standard_normal(1)
    ref = er.transform(X) @ w.T + b
    w2, b2 = S.fold_leace(w, b, er.proj, er.mu)
    np.testing.assert_allclose(X @ w2.T + b2, ref, atol=1e-10)


def test_metric_m():
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.standard_normal((10, 2)))
    D = rng.standard_normal((30, 10))
    assert S.metric_m(D, D, Q) == pytest.approx(0.0)
    assert S.metric_m(D - D @ Q @ Q.T, D, Q) == pytest.approx(1.0)
