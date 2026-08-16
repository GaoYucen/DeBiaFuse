"""Fair, leakage-safe baseline comparison on the Hongfu daily series."""
import argparse, json, time, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from debiafuse_pipeline import read_hongfu, chronological_split, TrainMinMaxScaler, make_windows


class DLinear(nn.Module):
    def __init__(self, look_back=24, horizon=6, kernel=5):
        super().__init__(); self.horizon = horizon
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=kernel // 2, count_include_pad=False)
        self.trend = nn.Linear(look_back, horizon); self.seasonal = nn.Linear(look_back, horizon)
    def forward(self, x):
        z = x.transpose(1, 2)
        trend = self.avg(z)
        if trend.shape[-1] != z.shape[-1]: trend = trend[..., :z.shape[-1]]
        seasonal = (z - trend)[:, 0, :]
        return (self.trend(trend[:, 0, :]) + self.seasonal(seasonal)).unsqueeze(-1)


class LSTM(nn.Module):
    def __init__(self, look_back=24, horizon=6):
        super().__init__(); self.rnn = nn.LSTM(1, 64, 2, batch_first=True, dropout=.1); self.head = nn.Linear(64, horizon)
    def forward(self, x): return self.head(self.rnn(x)[0][:, -1])


def result_base(file, model, n, ntr, nva, nte, look_back, horizon):
    return {"dataset": file, "model": model, "train_windows": ntr, "val_windows": nva,
            "test_windows": nte, "look_back": look_back, "horizon": horizon,
            "training_time": 0., "inference_time": 0., "parameter_count": 0,
            "status": "ok"}


def score(y, p):
    y, p = y.reshape(-1), p.reshape(-1); mse = mean_squared_error(y, p)
    return {"MAE": float(mean_absolute_error(y, p)), "RMSE": float(np.sqrt(mse)),
            "MSE": float(mse), "R2": float(r2_score(y, p))}


def fit_torch(kind, Xtr, Ytr, Xva, Yva, Xte, epochs, batch, device, look_back, horizon):
    model = LSTM(look_back, horizon) if kind == "lstm" else DLinear(look_back, horizon)
    model.to(device); opt = torch.optim.Adam(model.parameters(), 1e-3); loss_fn = nn.HuberLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr[..., None]), torch.from_numpy(Ytr)), batch_size=batch, shuffle=True)
    vx, vy = torch.from_numpy(Xva[..., None]).to(device).float(), torch.from_numpy(Yva).to(device).float()
    best, state, wait = float("inf"), None, 0; start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            opt.zero_grad(); pred = model(x.to(device).float()); pred = pred.squeeze(-1) if pred.ndim == 3 else pred; loss_fn(pred, y.to(device).float()).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(vx); vp = vp.squeeze(-1) if vp.ndim == 3 else vp; val = float(loss_fn(vp, vy))
        if val < best: best, wait, state = val, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 15: break
    model.load_state_dict(state); train_time = time.perf_counter() - start; model.eval(); start = time.perf_counter()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte[..., None]).to(device).float()); pred = pred.squeeze(-1) if pred.ndim == 3 else pred; pred = pred.cpu().numpy()
    return pred, train_time, time.perf_counter() - start, sum(p.numel() for p in model.parameters())


def arima_predict(train_values, n_test, horizon):
    from statsmodels.tsa.arima.model import ARIMA
    out = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(train_values, order=(1, 1, 1)).fit(method_kwargs={"maxiter": 25, "disp": 0})
        fc = np.asarray(fit.forecast(n_test + horizon - 1))
        return np.asarray([fc[i:i + horizon] for i in range(n_test)])
    except Exception:
        return np.repeat(train_values[-1], (n_test, horizon))


def crossformer_predict(Xtr, Ytr, Xva, Yva, Xte, epochs, batch, device, look_back, horizon):
    sys.path.insert(0, str(Path(__file__).parent / "crossformer" / "Crossformer"))
    from cross_models.cross_former import Crossformer
    model = Crossformer(data_dim=1, in_len=look_back, out_len=horizon, seg_len=6, win_size=2,
                        factor=5, d_model=64, d_ff=128, n_heads=2, e_layers=1, dropout=.1).to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3); loss_fn = nn.HuberLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr[..., None]), torch.from_numpy(Ytr)), batch_size=batch, shuffle=True)
    vx, vy = torch.from_numpy(Xva[..., None]).to(device).float(), torch.from_numpy(Yva).to(device).float()
    best, state, wait = float("inf"), None, 0; start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            opt.zero_grad(); p = model(x.to(device).float()).squeeze(-1); loss_fn(p, y.to(device).float()).backward(); opt.step()
        model.eval()
        with torch.no_grad(): val = float(loss_fn(model(vx).squeeze(-1), vy))
        if val < best: best, wait, state = val, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 15: break
    model.load_state_dict(state); train_time = time.perf_counter() - start; model.eval(); start = time.perf_counter()
    with torch.no_grad(): pred = model(torch.from_numpy(Xte[..., None]).to(device).float()).squeeze(-1).cpu().numpy()
    return pred, train_time, time.perf_counter() - start, sum(p.numel() for p in model.parameters())


def run_file(path, args):
    values, dates = read_hongfu(str(path)); split = chronological_split(values, dates)
    scaler = TrainMinMaxScaler().fit(split.train); z = scaler.transform(values)
    a, b = len(split.train), len(split.train) + len(split.val)
    Xtr, Ytr, itr = make_windows(z, args.look_back, args.horizon, 0, a)
    Xva, Yva, iva = make_windows(z, args.look_back, args.horizon, a, b)
    Xte, Yte, ite = make_windows(z, args.look_back, args.horizon, b, len(z))
    assert len(Xte) and np.all(ite >= b) and np.all(itr < a) and np.all(iva >= a) and np.all(iva + args.horizon - 1 < b)
    out = []
    def add(name, pred, tr=0., inf=0., params=0.):
        r = result_base(path.name, name, len(z), len(Xtr), len(Xva), len(Xte), args.look_back, args.horizon)
        r.update(score(scaler.inverse_transform(Yte), scaler.inverse_transform(pred)), training_time=tr, inference_time=inf, parameter_count=params); out.append(r)
    add("persistence", np.repeat(Xte[:, -1:], args.horizon, axis=1))
    for name in args.models:
        if name == "persistence": continue
        try:
            if name == "arima":
                t = time.perf_counter(); pred = arima_predict(z[:b], len(Xte), args.horizon); add(name, pred, 0., time.perf_counter()-t)
            elif name in ("lstm", "dlinear"):
                pred, tr, inf, n = fit_torch(name, Xtr, Ytr, Xva, Yva, Xte, args.epochs, args.batch_size, args.device, args.look_back, args.horizon); add(name, pred, tr, inf, n)
            elif name == "crossformer":
                pred, tr, inf, n = crossformer_predict(Xtr, Ytr, Xva, Yva, Xte, args.epochs, args.batch_size, args.device, args.look_back, args.horizon); add(name, pred, tr, inf, n)
            elif name == "uni2ts":
                raise RuntimeError("Uni2TS adapter requires GluonTS and compatible Moirai dependencies; not installed in current runtime")
        except Exception as e:
            r = result_base(path.name, name, len(z), len(Xtr), len(Xva), len(Xte), args.look_back, args.horizon); r.update(status="unavailable", error=f"{type(e).__name__}: {e}"); out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default="DLA/data/Hongfu/deflection"); ap.add_argument("--output", default="results/baselines_hongfu.json")
    ap.add_argument("--look-back", type=int, default=24); ap.add_argument("--horizon", type=int, default=6); ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default="persistence,arima,lstm,dlinear,crossformer,uni2ts"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); args = ap.parse_args(); args.models = [x.strip() for x in args.models.split(",")]
    torch.manual_seed(args.seed); np.random.seed(args.seed); rows = []
    for p in sorted(Path(args.data_dir).glob("*.xls")): rows.extend(run_file(p, args))
    if not rows: raise FileNotFoundError(args.data_dir)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8"); pd.DataFrame(rows).to_csv(Path(args.output).with_suffix(".csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__": main()
