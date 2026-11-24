import numpy as np
import torch

from fmritest.finetune import DiagnosisModel, evaluate, finetune, subject_logits
from fmritest.interp.api import Inspector
from fmritest.models.transformer import MaskedRegionTransformer

FCFG = {"steps": 30, "batch_size": 16, "lr_head": 1e-2, "lr_backbone": 1e-3, "weight_decay": 0.0}


def _data(n=40, T=96, R=10, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 0, 1, 1], n // 4)  # every split below gets both classes
    site = np.repeat([0, 1], n // 2)
    X = rng.standard_normal((n, T, R))
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True) + 1.0 * y[:, None, None]  # mean shift
    return X.astype(np.float64), y, site


def test_head_learns_and_backbone_untouched():
    torch.manual_seed(0)
    bb = MaskedRegionTransformer(10, 32, d_model=16, n_heads=2, n_layers=1, d_mlp=32)
    before = {k: v.clone() for k, v in bb.state_dict().items()}
    X, y, site = _data()
    tr = np.arange(0, 40, 2)
    m, info = finetune(bb, X, y, site, tr, dict(FCFG, steps=200), seed=0, n_threads=1)
    for k, v in bb.state_dict().items():  # original backbone is not modified in place
        torch.testing.assert_close(v, before[k])
    sp = {"train": tr, "test": np.arange(1, 40, 2), "test_balanced": np.arange(1, 40, 2),
          "test_site": np.arange(1, 40, 2)}
    r = evaluate(m, X, y, sp)
    assert r["test"]["auc"] > 0.9 and "shortcut_gap_bacc" in r
    assert subject_logits(m, X[:3]).shape == (3,)


def test_reweight_runs_and_inspector_compatible(tmp_path):
    torch.manual_seed(0)
    bb = MaskedRegionTransformer(10, 32, d_model=16, n_heads=2, n_layers=1, d_mlp=32)
    X, y, site = _data()
    m, _ = finetune(bb, X, y, site, np.arange(40), FCFG, seed=1, n_threads=1, reweight=True)
    insp = Inspector(m, tmp_path / "l.jsonl")
    x = torch.randn(2, 10, 32)
    out, cache = insp.run_with_cache(x)
    assert out.shape == (2,) and cache["blocks.0.attn.hook_pattern"].shape == (2, 2, 10, 10)
    with insp.intervention():
        insp.ablate("blocks.0.hook_resid_post")
        assert not torch.allclose(insp.model(x), out)
    assert insp.is_pristine()
