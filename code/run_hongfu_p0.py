"""Run leakage-safe Hongfu baselines (P0).

Usage from repository root:
    python code/run_hongfu_p0.py --epochs 50
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from debiafuse_pipeline import chronological_split, read_hongfu, TrainMinMaxScaler, make_windows


class ForecastLSTM(nn.Module):
    def __init__(self, hidden=64, horizon=6):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        return self.head(self.rnn(x)[0][:, -1])


def metrics(y, p):
    mse = mean_squared_error(y.reshape(-1), p.reshape(-1))
    return {"MAE": float(mean_absolute_error(y.reshape(-1), p.reshape(-1))),
            "RMSE": float(np.sqrt(mse)), "MSE": float(mse),
            "R2": float(r2_score(y.reshape(-1), p.reshape(-1)))}


def fit_one(path, look_back, horizon, epochs, batch_size, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    values, dates = read_hongfu(str(path))
    split = chronological_split(values, dates)
    scaler = TrainMinMaxScaler().fit(split.train)
    z = scaler.transform(values)
    # target ranges are raw indices; test windows may use historical context.
    Xtr, Ytr, _ = make_windows(z, look_back, horizon, 0, len(split.train))
    Xva, Yva, _ = make_windows(z, look_back, horizon, len(split.train), len(split.train) + len(split.val))
    Xte, Yte, _ = make_windows(z, look_back, horizon, len(split.train) + len(split.val), len(values))
    if min(len(Xtr), len(Xva), len(Xte)) == 0:
        raise ValueError(f"Not enough samples in {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForecastLSTM(horizon=horizon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss()
    train = DataLoader(TensorDataset(torch.from_numpy(Xtr[..., None]), torch.from_numpy(Ytr)), batch_size=batch_size, shuffle=True)
    valx, valy = torch.from_numpy(Xva[..., None]).to(device), torch.from_numpy(Yva).to(device)
    best, best_state, patience = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for x, y in train:
            x, y = x.to(device).float(), y.to(device).float()
            opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
        model.eval()
        with torch.no_grad(): val_loss = float(loss_fn(model(valx.float()), valy.float()))
        if val_loss < best:
            best, patience = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 15: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): pred_z = model(torch.from_numpy(Xte[..., None]).to(device).float()).cpu().numpy()
    pred = scaler.inverse_transform(pred_z); true = scaler.inverse_transform(Yte)
    persistence = np.repeat(Xte[:, -1:,], horizon, axis=1)
    persistence = scaler.inverse_transform(persistence)
    return {"file": path.name, "n": len(values), "train_windows": len(Xtr), "val_windows": len(Xva),
            "test_windows": len(Xte), "device": str(device), "lstm": metrics(true, pred),
            "persistence": metrics(true, persistence)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="DLA/data/Hongfu/deflection")
    ap.add_argument("--look-back", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="log/hongfu_p0.json")
    args = ap.parse_args()
    start = time.time()
    results = [fit_one(p, args.look_back, args.horizon, args.epochs, args.batch_size, args.seed)
               for p in sorted(Path(args.data_dir).glob("*.xls"))]
    if not results: raise FileNotFoundError(f"No .xls files under {args.data_dir}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"results": results, "seconds": time.time() - start}, indent=2), encoding="utf-8")
    for r in results: print(r["file"], "LSTM", r["lstm"], "Persistence", r["persistence"])


if __name__ == "__main__": main()
