import json

import numpy as np
import pytest
import torch

from fmritest.interp.api import Inspector, state_hash
from fmritest.models.transformer import MaskedRegionTransformer


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(0)
    m = MaskedRegionTransformer(10, 16, d_model=32, n_heads=4, n_layers=2, d_mlp=64).eval()
    x = torch.randn(4, 10, 16)
    mask = torch.zeros(4, 10, dtype=torch.bool)
    mask[:, 0] = True
    insp = Inspector(m, tmp_path / "log.jsonl", config={"a": 1}, seed=7, ref_batch=(x, mask))
    return insp, x, mask, tmp_path / "log.jsonl"


def test_cache_shapes(setup):
    insp, x, mask, _ = setup
    _, c = insp.run_with_cache(x, mask)
    assert c["blocks.0.attn.hook_pattern"].shape == (4, 4, 10, 10)
    assert c["blocks.1.attn.hook_z"].shape == (4, 10, 4, 8)
    assert c["blocks.1.mlp.hook_post"].shape == (4, 10, 64)
    torch.testing.assert_close(c["blocks.0.attn.hook_pattern"].sum(-1), torch.ones(4, 4, 10))
    assert insp.head_weights(1, 2)["W_O"].shape == (32, 8)


def test_ablation_changes_output_and_restore_is_exact(setup):
    insp, x, mask, log = setup
    base = insp.model(x, mask).detach().clone()
    h0 = state_hash(insp.model)
    with insp.intervention():
        insp.ablate("blocks.0.attn.hook_z", head=1)
        insp.ablate("blocks.1.mlp.hook_post", neuron=3)
        insp.project_out_weight("blocks.0.attn.W_O.weight", np.ones(32), side="out")
        assert not torch.allclose(insp.model(x, mask), base)
    assert insp.is_pristine() and state_hash(insp.model) == h0
    torch.testing.assert_close(insp.model(x, mask), base, rtol=0, atol=0)
    entries = [json.loads(l) for l in open(log)]
    assert [e["intervention"]["type"] for e in entries] == ["ablate", "ablate", "project_out_weight", "restore"]
    assert all(e["seed"] == 7 and e["config"] == {"a": 1} for e in entries)
    w = entries[2]["diff"]["blocks.0.attn.W_O.weight"]
    assert w["fro_delta"] > 0 and entries[2]["diff"]["ref_output_mse_change"] > 0


def test_restore_on_exception(setup):
    insp, x, mask, _ = setup
    with pytest.raises(RuntimeError):
        with insp.intervention():
            insp.set_weight("unembed.bias", np.zeros(16))
            raise RuntimeError("boom")
    assert insp.is_pristine()


def test_patch_with_own_activation_is_identity(setup):
    insp, x, mask, _ = setup
    base, cache = insp.run_with_cache(x, mask)
    with insp.intervention():
        insp.patch("blocks.0.hook_resid_post", cache["blocks.0.hook_resid_post"])
        torch.testing.assert_close(insp.model(x, mask), base)
    # patching from a different input changes output, and a single head patch is local
    _, other = insp.run_with_cache(torch.randn_like(x), mask)
    with insp.intervention():
        insp.patch("blocks.1.attn.hook_z", other["blocks.1.attn.hook_z"], head=0)
        _, c = insp.run_with_cache(x, mask)
        z = c["blocks.1.attn.hook_z"]
        torch.testing.assert_close(z[:, :, 0], other["blocks.1.attn.hook_z"][:, :, 0])
        torch.testing.assert_close(z[:, :, 1:], cache["blocks.1.attn.hook_z"][:, :, 1:])


def test_direction_removal(setup):
    insp, x, mask, _ = setup
    v = np.random.default_rng(0).standard_normal(32)
    with insp.intervention():
        insp.remove_direction("blocks.0.hook_resid_post", v)
        _, c = insp.run_with_cache(x, mask, ["blocks.0.hook_resid_post"])
        vt = torch.tensor(v / np.linalg.norm(v), dtype=torch.float32)
        assert float((c["blocks.0.hook_resid_post"] @ vt).abs().max()) < 1e-5
    W0 = insp.weights("blocks.0.mlp.W_out.weight")
    with insp.intervention():
        insp.project_out_weight("blocks.0.mlp.W_out.weight", v, side="out")
        W = insp.weights("blocks.0.mlp.W_out.weight")
        b = insp.weights("blocks.0.mlp.W_out.bias")
        assert float((vt @ W).abs().max()) < 1e-5 and abs(float(vt @ b)) < 1e-5
    torch.testing.assert_close(insp.weights("blocks.0.mlp.W_out.weight"), W0)


def test_mean_ablation_positions(setup):
    insp, x, mask, _ = setup
    with insp.intervention():
        insp.ablate("blocks.0.hook_resid_post", kind="mean", mean_value=np.full(32, 0.5), positions=[2, 3])
        _, c = insp.run_with_cache(x, mask, ["blocks.0.hook_resid_post"])
        r = c["blocks.0.hook_resid_post"]
        assert torch.all(r[:, [2, 3]] == 0.5) and not torch.all(r[:, 4] == 0.5)
    with pytest.raises(ValueError):
        insp.ablate("blocks.0.hook_resid_post", kind="mean")


def test_remove_direction_head_only_on_hook_z(setup):
    insp, x, mask, _ = setup
    with pytest.raises(ValueError):
        insp.remove_direction("blocks.0.attn.hook_pattern", np.ones(10), head=0)
    insp.remove_direction("blocks.0.attn.hook_z", np.ones(8), head=0)
    insp.restore()
