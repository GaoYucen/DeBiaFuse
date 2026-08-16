import numpy as np
import torch

from debiafuse_pipeline import (
    chronological_split, TrainMinMaxScaler, make_windows,
    causal_moving_average, JointHighFrequencyModel,
    make_decomposed_windows,
)


def test_split_is_chronological_and_disjoint():
    x = np.arange(20, dtype=np.float32)
    d = np.arange("2020-01-01", "2020-01-21", dtype="datetime64[D]")
    s = chronological_split(x, d)
    assert s.train_dates[-1] < s.val_dates[0] < s.test_dates[0]
    assert len(set(s.train_dates) & set(s.test_dates)) == 0


def test_windows_are_assigned_by_target_timestamp():
    x = np.arange(30, dtype=np.float32)
    X, Y, starts = make_windows(x, 4, 3, target_start=20, target_end=30)
    assert starts[0] == 20
    assert np.array_equal(X[0], [16, 17, 18, 19])
    assert np.array_equal(Y[0], [20, 21, 22])
    assert np.all(starts + 2 < 30)


def test_scaler_is_train_only():
    scaler = TrainMinMaxScaler().fit([0, 1, 2])
    assert np.allclose(scaler.transform([0, 2]), [0, 1])
    assert np.allclose(scaler.inverse_transform([0.5]), [1])


def test_causal_preprocessing_is_future_invariant():
    x = np.arange(20, dtype=np.float32)
    y = x.copy(); y[12:] += 10000
    assert np.allclose(causal_moving_average(x, 5)[:12], causal_moving_average(y, 5)[:12])


def test_joint_biaxial_shapes_and_mask():
    model = JointHighFrequencyModel(3, 8, 4, d_model=16, n_heads=4, depth=1)
    pred = model(torch.randn(2, 8, 3), torch.tensor([[1., 1., 0.], [1., 0., 0.]]))
    assert pred.shape == (2, 4, 3)


def test_window_local_decomposition_shapes():
    x = np.sin(np.arange(40, dtype=np.float32) / 3)
    xl, xh, yl, yh, mask, starts = make_decomposed_windows(x, 8, 4, n_high=2, target_start=12, target_end=30)
    assert xl.shape[1:] == (8,)
    assert xh.shape[1:] == (8, 3)  # 2 IMF slots + residual
    assert yl.shape[1:] == (4,)
    assert yh.shape[1:] == (4, 3)
    assert mask.shape[1:] == (3,)  # IMF slots + residual
    assert starts[0] == 12
