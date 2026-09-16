"""Mechanical protocol repair for BRIDGE-V3-01.

The first V3 runner accidentally added gradient clipping that was absent from the
frozen R3B global-only control. This wrapper restores the frozen optimizer/training
semantics without changing the pre-authorized V0/V1/V2/V3 architectures, data,
split, loss, metric, horizon, or model-selection policy.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import torch

BASE = Path(__file__).with_name('bridge_v3_r1.py')

# Frozen R3B training did not apply gradient clipping. The original V3 runner calls
# clip_grad_norm_ once per optimizer step; neutralize only that accidental drift.
def _no_clip(parameters, max_norm, *args, **kwargs):
    return torch.tensor(0.0)

torch.nn.utils.clip_grad_norm_ = _no_clip

spec = importlib.util.spec_from_file_location('bridge_v3_r1_base', BASE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if __name__ == '__main__':
    mod.main()
