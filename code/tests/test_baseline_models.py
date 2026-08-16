import torch
from run_baselines import DLinear, LSTM


def test_dlinear_shape():
    assert DLinear(24, 6)(torch.randn(3, 24, 1)).shape == (3, 6, 1)


def test_lstm_shape():
    assert LSTM(24, 6)(torch.randn(3, 24, 1)).shape == (3, 6)
