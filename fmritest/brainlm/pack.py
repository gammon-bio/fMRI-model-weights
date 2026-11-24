"""Pack a parcel x time matrix into BrainLM's 3 x 432 x 432 frame, as in the BrainLM tutorial.

Source: BrainLM repo, toolkit/BrainLM_Tutorial.ipynb
  - cell `preprocess_images` (notebook JSON lines ~684-708): the recording is a [num_voxels=424,
    length=200] matrix (parcels = rows = image HEIGHT, time = columns = image WIDTH), divided by
    `max_val_to_scale = 5.6430855`, then repeated over 3 channels -> [3, 424, 200].
  - cell "Save CLS Tokens + All Embeddings" (lines ~740-770): F.pad(..., (w_half, w_half, h_half, h_half),
    "constant", -1) to [3, 432, 432], i.e. 4 rows above/below and 116 columns left/right, pad value -1.
  - cell lines ~818-832: the data-carrying patch tokens are rows vert_token_start:vert_token_end and
    columns hor_token_start:hor_token_end of the 27 x 27 grid (start = pad // 16, end = start +
    ceil(n / 16)); this module's `data_patch_mask` reproduces that.
Robust scaling follows toolkit/BrainLM_Toolkit.py ("Voxelwise Robust Scaler Normalization", ~l.296):
per-recording temporal demeaning, then per-parcel (x - median) / IQR with median / IQR from the
TRAINING recordings. The toolkit line is written `(x - median / IQR)` (operator precedence); we use
the evidently intended `(x - median) / IQR`. The tutorial also reorders A424 parcels by y-coordinate;
synthetic parcels have no coordinates (network membership is randomly permuted over regions), so no
reordering is applied.
Time therefore runs along the LAST image axis (width), and so along the LAST axis of the patch
kernel `projection.weight` [768, 3, 16(parcel rows), 16(time)].
"""
from __future__ import annotations

import numpy as np

IMAGE = 432
PATCH = 16
GRID = IMAGE // PATCH          # 27
MAX_VAL_TO_SCALE = 5.6430855   # tutorial constant
PAD_VALUE = -1.0
TIME_AXIS_IMAGE = -1           # width
KERNEL_TIME_AXIS = -1          # projection.weight[..., time]


def fit_robust_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X_train (N, T, R) -> per-parcel median and IQR (R,) of temporally demeaned training data."""
    Xd = X_train - X_train.mean(axis=1, keepdims=True)
    flat = Xd.transpose(2, 0, 1).reshape(X_train.shape[2], -1)
    q25, med, q75 = np.percentile(flat, [25, 50, 75], axis=1)
    iqr = q75 - q25
    return med, np.where(iqr > 0, iqr, 1.0)


def robust_scale(X: np.ndarray, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    """(N, T, R) -> (N, T, R) scaled (per-recording demean, then (x - med) / iqr)."""
    Xd = X - X.mean(axis=1, keepdims=True)
    return (Xd - med) / iqr


def pad_offsets(n_parcels: int, n_time: int) -> tuple[int, int]:
    """(top, left) padding; the tutorial pads symmetrically with total // 2 on each side."""
    return (IMAGE - n_parcels) // 2, (IMAGE - n_time) // 2


def pack(Xs: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Scaled recordings (N, T, R) -> pixel values (N, 3, 432, 432): rows = parcels, columns = time."""
    N, T, R = Xs.shape
    top, left = pad_offsets(R, T)
    if R + 2 * top > IMAGE or T + 2 * left > IMAGE:
        raise ValueError("recording does not fit the 432 x 432 frame")
    out = np.full((N, 3, IMAGE, IMAGE), PAD_VALUE, dtype=dtype)
    img = (Xs.transpose(0, 2, 1) / MAX_VAL_TO_SCALE).astype(dtype)  # (N, R, T)
    out[:, :, top:top + R, left:left + T] = img[:, None]
    return out


def data_patch_mask(n_parcels: int = 424, n_time: int = 200) -> np.ndarray:
    """(27, 27) boolean grid, True for patches holding real data (tutorial token-range formula)."""
    top, left = pad_offsets(n_parcels, n_time)
    r0, c0 = top // PATCH, left // PATCH
    r1 = r0 + int(np.ceil(n_parcels / PATCH))
    c1 = c0 + int(np.ceil(n_time / PATCH))
    m = np.zeros((GRID, GRID), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m
