import numpy as np
import torch

from fmritest.models.train import make_eval_set, to_windows
from fmritest.models.transformer import MaskedRegionTransformer, masked_mse


def test_forward_shape_and_mask_effect():
    torch.manual_seed(0)
    m = MaskedRegionTransformer(30, 48).eval()
    x = torch.randn(3, 30, 48)
    mask = torch.zeros(3, 30, dtype=torch.bool)
    mask[:, :5] = True
    out = m(x, mask)
    assert out.shape == x.shape
    # masked content cannot influence the output
    x2 = x.clone()
    x2[:, :5] = torch.randn(3, 5, 48)
    torch.testing.assert_close(m(x2, mask), out)


def test_masked_mse_only_counts_masked_tokens():
    pred = torch.zeros(2, 4, 3, requires_grad=True)
    target = torch.ones(2, 4, 3)
    target[:, 1:] = 100.0
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[:, 0] = True
    loss = masked_mse(pred, target, mask)
    assert float(loss.detach()) == 1.0
    loss.backward()
    assert torch.all(pred.grad[:, 1:] == 0)


def test_windows_and_eval_set_deterministic():
    X = np.random.default_rng(0).standard_normal((4, 200, 30))
    w = to_windows(X, 48)
    assert w.shape == (4, 4, 30, 48)
    np.testing.assert_array_equal(w[1, 2, 5], X[1, 96:144, 5])
    a, b = make_eval_set(X, 48, 2, 0.2, 1), make_eval_set(X, 48, 2, 0.2, 1)
    np.testing.assert_array_equal(a[1], b[1])
    assert a[1].sum(1).tolist() == [6] * len(a[1])
