import torch
from torch import nn
from debiafuse_pipeline import JointHighFrequencyModel


class DeBiaFuseV2(nn.Module):
    """Leakage-safe causal trend + joint high-frequency predictor."""
    def __init__(self, n_components, look_back, horizon, hidden=64):
        super().__init__()
        self.horizon = horizon
        self.low = nn.LSTM(1, hidden, 2, batch_first=True, dropout=.1)
        self.low_head = nn.Linear(hidden, horizon)
        self.high = JointHighFrequencyModel(n_components, look_back, horizon, d_model=64, n_heads=4, depth=2)

    def forward(self, low_x, high_x, component_mask=None):
        low_pred = self.low_head(self.low(low_x.unsqueeze(-1))[0][:, -1])
        high_pred = self.high(high_x, component_mask)
        return low_pred, high_pred, low_pred + high_pred.sum(-1)


def debiafuse_loss(low_pred, high_pred, total_pred, low_y, high_y, total_y, component_scale, total_scale, alpha=.4, beta=.6):
    scale = torch.as_tensor(component_scale, dtype=high_pred.dtype, device=high_pred.device).clamp_min(1e-6)
    gscale = torch.as_tensor(total_scale, dtype=high_pred.dtype, device=high_pred.device).clamp_min(1e-6)
    component = nn.functional.huber_loss(low_pred / gscale, low_y / gscale) + nn.functional.huber_loss(high_pred / scale, high_y / scale)
    global_loss = nn.functional.huber_loss(total_pred / gscale, total_y / gscale)
    return alpha * component + beta * global_loss
