"""Leakage-safe data and joint multi-component modelling utilities."""
from dataclasses import dataclass
from typing import Iterable, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
try:
    from PyEMD import EMD
except ImportError:  # keep data/attention utilities usable for baseline-only installs
    EMD = None


@dataclass
class TimeSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_dates: np.ndarray
    val_dates: np.ndarray
    test_dates: np.ndarray


def read_hongfu(path: str, return_mask=False):
    df = pd.read_excel(path, engine="xlrd")
    required = {"时间", "测量的桥面系挠度值"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    dates = pd.to_datetime(df["时间"].astype(str).str[:10], errors="coerce")
    values = pd.to_numeric(df["测量的桥面系挠度值"], errors="coerce")
    clean = pd.DataFrame({"date": dates, "value": values}).dropna()
    daily = clean.groupby("date", sort=True)["value"].mean()
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full)
    missing = daily.isna().to_numpy()
    daily = daily.ffill().bfill()
    out = (daily.to_numpy(dtype=np.float32), full.to_numpy(dtype="datetime64[ns]"))
    return (*out, missing.astype(np.float32)) if return_mask else out


def hongfu_quality_report(path: str):
    df = pd.read_excel(path, engine="xlrd")
    raw_dates = pd.to_datetime(df["时间"].astype(str).str[:10], errors="coerce")
    vals = pd.to_numeric(df["测量的桥面系挠度值"], errors="coerce")
    valid = pd.DataFrame({"date": raw_dates, "value": vals}).dropna(subset=["date"])
    daily = valid.groupby("date")["value"].mean().sort_index()
    expected = pd.date_range(daily.index.min(), daily.index.max(), freq="D") if len(daily) else pd.DatetimeIndex([])
    return {"dataset": Path(path).name, "date_start": str(daily.index.min().date()) if len(daily) else "",
            "date_end": str(daily.index.max().date()) if len(daily) else "", "expected_days": len(expected),
            "observed_days": len(daily), "missing_days": len(expected.difference(daily.index)),
            "missing_ratio": len(expected.difference(daily.index)) / max(1, len(expected)),
            "duplicate_raw_dates": int(raw_dates.dropna().duplicated().sum()), "nan_raw_values": int(vals.isna().sum())}


def chronological_split(values, dates, train_ratio=0.7, val_ratio=0.1) -> TimeSplit:
    values, dates = np.asarray(values), np.asarray(dates)
    if len(values) != len(dates) or len(values) < 3:
        raise ValueError("values and dates must have equal length >= 3")
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Invalid split ratios")
    n_train = max(1, int(len(values) * train_ratio))
    n_val = max(1, int(len(values) * val_ratio))
    if n_train + n_val >= len(values):
        n_val = len(values) - n_train - 1
    out = TimeSplit(values[:n_train], values[n_train:n_train+n_val], values[n_train+n_val:],
                    dates[:n_train], dates[n_train:n_train+n_val], dates[n_train+n_val:])
    assert out.train_dates[-1] < out.val_dates[0]
    assert out.val_dates[-1] < out.test_dates[0]
    return out


class TrainMinMaxScaler:
    def __init__(self):
        self.lo = self.hi = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        self.lo, self.hi = float(np.nanmin(x)), float(np.nanmax(x))
        return self

    def transform(self, x):
        if self.lo is None:
            raise RuntimeError("fit must be called on training data first")
        den = self.hi - self.lo
        return np.zeros_like(np.asarray(x, dtype=np.float32)) if den == 0 else (np.asarray(x, dtype=np.float32) - self.lo) / den

    def inverse_transform(self, x):
        if self.lo is None:
            raise RuntimeError("fit must be called first")
        return np.asarray(x, dtype=np.float32) * (self.hi - self.lo) + self.lo


def make_windows(values, look_back, horizon, target_start=0, target_end=None):
    values = np.asarray(values, dtype=np.float32)
    target_end = len(values) if target_end is None else min(target_end, len(values))
    X, Y, target_indices = [], [], []
    for t in range(max(look_back, target_start), target_end - horizon + 1):
        X.append(values[t-look_back:t])
        Y.append(values[t:t+horizon])
        target_indices.append(t)
    return np.asarray(X), np.asarray(Y), np.asarray(target_indices, dtype=np.int64)


def causal_moving_average(x, window):
    x = np.asarray(x, dtype=np.float32)
    return pd.Series(x).rolling(window, min_periods=1).mean().to_numpy(dtype=np.float32)


def spectral_features(component, fs=1.0):
    x = np.asarray(component, dtype=np.float32)
    spectrum = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    if len(spectrum) > 1:
        spectrum[0] = 0
    energy = float(spectrum.sum())
    if energy <= 1e-12:
        return np.array([0., 0., 0., 0.], dtype=np.float32)
    centroid = float((freqs * spectrum).sum() / energy)
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * spectrum).sum() / energy))
    return np.array([float(freqs[int(np.argmax(spectrum))]), centroid, bandwidth, energy], dtype=np.float32)


def window_local_emd(history, n_high=3, trend_window=30):
    """Decompose one historical window only; output fixed K high components + residual."""
    history = np.asarray(history, dtype=np.float32)
    if len(history) < 4:
        out = np.zeros((n_high + 1, len(history)), dtype=np.float32)
        out[-1] = history
        return history.copy(), out, np.zeros(n_high, dtype=np.float32)
    trend = causal_moving_average(history, min(trend_window, len(history)))
    residual = history - trend
    imfs = EMD()(residual) if EMD is not None else []
    comps = [] if imfs is None else [np.asarray(v, dtype=np.float32) for v in imfs]
    comps.sort(key=lambda c: spectral_features(c)[1], reverse=True)
    high = comps[:n_high]
    used = np.sum(high, axis=0) if high else np.zeros_like(history)
    remainder = residual - used
    out = np.zeros((n_high + 1, len(history)), dtype=np.float32)
    mask = np.zeros(n_high, dtype=np.float32)
    for i, c in enumerate(high):
        out[i] = c
        mask[i] = 1.0
    out[-1] = remainder
    return trend, out, mask


def make_decomposed_windows(values, look_back, horizon, n_high=3,
                            target_start=0, target_end=None,
                            decomp_context=90, trend_window=30):
    """Create causal decomposition tensors for forecasting windows.

    For every input and target timestamp, decomposition sees only the prefix
    ending at that timestamp. This is slower than offline EMD, but is the
    reference implementation used for leakage checks and small experiments.
    """
    values = np.asarray(values, dtype=np.float32)
    target_end = len(values) if target_end is None else min(target_end, len(values))
    X_low, X_hf, Y_low, Y_hf, masks, starts = [], [], [], [], [], []
    cache = {}
    def at(idx):
        if idx not in cache:
            lo, hi, ma = window_local_emd(values[max(0, idx - decomp_context + 1):idx + 1], n_high, trend_window)
            cache[idx] = (lo[-1], hi[:, -1], ma)
        return cache[idx]
    for t in range(max(look_back, target_start), target_end - horizon + 1):
        low_hist, high_hist = [], []
        for idx in range(t - look_back, t):
            low, high, _ = at(idx)
            low_hist.append(low); high_hist.append(high)
        low_targets, high_targets = [], []
        for idx in range(t, t + horizon):
            low, high, mask = at(idx)
            low_targets.append(low); high_targets.append(high)
        # high includes K IMF slots and a residual slot; mask applies to IMF slots.
        X_low.append(low_hist); X_hf.append(np.asarray(high_hist))
        Y_low.append(low_targets); Y_hf.append(np.asarray(high_targets))
        # mask is historical only: use the forecast-origin predecessor.
        _, _, hist_mask = at(t - 1)
        masks.append(np.concatenate([hist_mask, np.ones(1, dtype=np.float32)])); starts.append(t)
    return (np.asarray(X_low, dtype=np.float32), np.asarray(X_hf, dtype=np.float32),
            np.asarray(Y_low, dtype=np.float32), np.asarray(Y_hf, dtype=np.float32),
            np.asarray(masks, dtype=np.float32), np.asarray(starts, dtype=np.int64))


class FactorizedBiaxialAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.temporal = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.component = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, component_mask=None):  # [B, L, N, D]
        b, l, n, d = x.shape
        t = x.permute(0, 2, 1, 3).reshape(b * n, l, d)
        t, _ = self.temporal(t, t, t)
        t = self.norm1(t).reshape(b, n, l, d).permute(0, 2, 1, 3)
        c = t.reshape(b * l, n, d)
        key_mask = None
        if component_mask is not None:
            cm = component_mask[:, None, :].expand(b, l, n).reshape(b * l, n) < 0.5
            key_mask = cm
        c, _ = self.component(c, c, c, key_padding_mask=key_mask)
        out = self.norm2(c).reshape(b, l, n, d)
        if component_mask is not None:
            out = out * component_mask[:, None, :, None]
        return out


class JointHighFrequencyModel(nn.Module):
    def __init__(self, n_components, look_back, horizon, d_model=64, n_heads=4, depth=2):
        super().__init__()
        self.n_components, self.horizon = n_components, horizon
        self.value = nn.Linear(1, d_model)
        self.component_embedding = nn.Parameter(torch.randn(1, 1, n_components, d_model) * 0.02)
        self.blocks = nn.ModuleList([FactorizedBiaxialAttention(d_model, n_heads) for _ in range(depth)])
        self.head = nn.Linear(look_back * d_model, horizon)

    def forward(self, x, component_mask=None):  # [B, L, N]
        h = self.value(x.unsqueeze(-1)) + self.component_embedding
        if component_mask is not None:
            h = h * component_mask[:, None, :, None]
        for block in self.blocks:
            h = h + block(h, component_mask)
            if component_mask is not None:
                h = h * component_mask[:, None, :, None]
        if component_mask is not None:
            h = h * component_mask[:, None, :, None]
        h = h.permute(0, 2, 1, 3).reshape(x.shape[0], self.n_components, -1)
        return self.head(h).permute(0, 2, 1)  # [B, horizon, N]


def robust_component_global_loss(pred, target, scale, alpha=0.4, beta=0.6):
    scale = torch.as_tensor(scale, dtype=pred.dtype, device=pred.device).clamp_min(1e-6)
    comp = nn.functional.huber_loss(pred / scale, target / scale)
    global_loss = nn.functional.huber_loss(pred.sum(-1), target.sum(-1))
    return alpha * comp + beta * global_loss
