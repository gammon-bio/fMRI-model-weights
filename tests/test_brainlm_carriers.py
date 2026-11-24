"""BrainLM Stage 0 packing and carrier-projection tests (no weights, no transformers needed)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fmritest.brainlm import carriers as C
from fmritest.brainlm import pack as P


def test_pack_shape_padding_and_mask():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 200, 424))
    med, iqr = P.fit_robust_scaler(X)
    Xs = P.robust_scale(X, med, iqr)
    px = P.pack(Xs)
    assert px.shape == (3, 3, 432, 432) and px.dtype == np.float32
    # 3 identical channels; rows = parcels (4 pad rows each side), columns = time (116 pad cols each side)
    assert np.array_equal(px[:, 0], px[:, 1]) and np.array_equal(px[:, 0], px[:, 2])
    assert P.pad_offsets(424, 200) == (4, 116)
    assert np.all(px[:, :, :4] == -1) and np.all(px[:, :, 428:] == -1)
    assert np.all(px[:, :, :, :116] == -1) and np.all(px[:, :, :, 316:] == -1)
    np.testing.assert_allclose(px[1, 0, 4 + 7, 116:316], Xs[1, :, 7] / P.MAX_VAL_TO_SCALE, rtol=1e-6)
    m = P.data_patch_mask(424, 200)
    assert m.shape == (27, 27) and m.sum() == 27 * 13
    assert m[:, 7:20].all() and not m[:, :7].any() and not m[:, 20:].any()
    # every patch with any real pixel is in the mask, and no masked-in patch is pure padding
    real = np.zeros((432, 432), bool)
    real[4:428, 116:316] = True
    has_real = real.reshape(27, 16, 27, 16).any(axis=(1, 3))
    assert np.array_equal(has_real, m)


def test_robust_scaler_uses_training_stats_per_parcel():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 50, 5)) * np.arange(1, 6)
    med, iqr = P.fit_robust_scaler(X)
    np.testing.assert_allclose(iqr / iqr[0], np.arange(1, 6), rtol=0.15)
    Xs = P.robust_scale(X, med, iqr)
    np.testing.assert_allclose(np.median(Xs.transpose(2, 0, 1).reshape(5, -1), axis=1), 0, atol=1e-9)


def test_carrier_basis_orthonormal():
    Q = C.carrier_basis([0.2, 1 / 3, 7 / 15, 0.6])
    assert Q.shape == (16, 8)
    np.testing.assert_allclose(Q.T @ Q, np.eye(8), atol=1e-10)
    R = C.random_basis(np.random.default_rng(0), 8)
    np.testing.assert_allclose(R.T @ R, np.eye(8), atol=1e-10)
    assert C.subspace_overlap(Q, Q) == pytest.approx(1.0)


def test_projector_idempotent_and_removes_in_band_keeps_orthogonal():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((6, 3, 16, 16))
    Q = C.carrier_basis([0.333])
    W1 = C.project_out(W, Q)
    np.testing.assert_allclose(C.project_out(W1, Q), W1, atol=1e-12)          # idempotent
    t = np.arange(16) * C.TR
    s_in = np.sin(2 * np.pi * 0.333 * t + 0.7)                                # pure in-band sinusoid
    np.testing.assert_allclose(W1 @ s_in, 0, atol=1e-10)                     # removed on the time axis
    v = rng.standard_normal(16)
    v_orth = v - Q @ (Q.T @ v)                                                # orthogonal to the carrier
    np.testing.assert_allclose(W1 @ v_orth, W @ v_orth, atol=1e-10)          # kept
    # torch path agrees with numpy
    Wt = C.project_out(torch.from_numpy(W), Q)
    np.testing.assert_allclose(Wt.numpy(), W1, atol=1e-12)


def test_correct_axis_on_synthetic_kernel_and_packed_image():
    """A carrier along the image's TIME axis (width) vanishes from the conv embedding after projecting
    along the kernel's last axis, and does not vanish when projecting along the parcel axis."""
    rng = np.random.default_rng(3)
    f = 0.467
    Q = C.carrier_basis([f])
    t = np.arange(200) * C.TR
    X = np.sin(2 * np.pi * f * t + 0.3)[None, :, None] * rng.standard_normal(424)[None, None, :]
    px = torch.from_numpy(P.pack(X, dtype=np.float64))                # carrier lives on the time (width) axis
    W = torch.from_numpy(rng.standard_normal((8, 3, 16, 16)))
    mask = P.data_patch_mask()
    # interior patches only: columns 8..18 are fully inside the data (edge patches straddle padding)
    interior = np.zeros_like(mask)
    interior[1:26, 8:19] = True

    def emb(Wk):
        return F.conv2d(px, Wk, stride=16)[0].permute(1, 2, 0).numpy()[interior]  # (n_patch, 8)

    base = np.abs(emb(W)).mean()
    right = np.abs(emb(C.project_out(W, Q, axis=P.KERNEL_TIME_AXIS))).max()
    wrong = np.abs(emb(C.project_out(W, Q, axis=-2))).mean()
    assert P.KERNEL_TIME_AXIS == -1
    assert right < 1e-9 * base
    assert wrong > 0.3 * base


def test_band_basis_and_bin_carriers():
    B = C.band_basis(2)
    assert B.shape == (16, 13)
    np.testing.assert_allclose(B.T @ B, np.eye(13), atol=1e-10)
    freqs = [C.dft_bin_freq(k) for k in range(2, 8)]
    np.testing.assert_allclose(freqs, [0.1736, 0.2604, 0.3472, 0.4340, 0.5208, 0.6076], atol=1e-4)
    Qc = C.carrier_basis(freqs[:4])
    # bin carriers lie inside the band, and distinct bins are orthogonal
    assert C.subspace_overlap(Qc, B) == pytest.approx(1.0)
    assert C.subspace_overlap(C.carrier_basis(freqs[:2]), C.carrier_basis(freqs[2:])) == pytest.approx(0, abs=1e-12)
    R = C.random_in_span(np.random.default_rng(0), B, 8)
    np.testing.assert_allclose(R.T @ R, np.eye(8), atol=1e-10)
    assert C.subspace_overlap(R, B) == pytest.approx(1.0)


def test_carrier_share():
    rng = np.random.default_rng(4)
    W = rng.standard_normal((5, 3, 16, 16))
    Qc = C.carrier_basis([C.dft_bin_freq(3), C.dft_bin_freq(5)])
    assert C.carrier_share(C.project_out(W, Qc) - W, Qc) == pytest.approx(1.0)
    Qo = C.carrier_basis([C.dft_bin_freq(2)])
    assert C.carrier_share(C.project_out(W, Qo) - W, Qc) == pytest.approx(0.0, abs=1e-12)


def test_group_balanced_order():
    from fmritest.brainlm.model import group_balanced_order
    g = np.array([0] * 90 + [1] * 10)
    o = group_balanced_order(np.random.default_rng(0), g)
    assert len(o) == 100 and set(np.unique(g[o])) == {0, 1}
    big = group_balanced_order(np.random.default_rng(1), np.repeat(g, 50))
    assert abs(np.mean(np.repeat(g, 50)[big]) - 0.5) < 0.03
