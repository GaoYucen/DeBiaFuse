"""Train leakage-safe DeBiaFuse v2 on Hongfu."""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from debiafuse_pipeline import (read_hongfu, chronological_split, TrainMinMaxScaler,
                                make_decomposed_windows, hongfu_quality_report)
from models.debiafuse_v2 import DeBiaFuseV2, debiafuse_loss


def score(y, p):
    y, p = y.reshape(-1), p.reshape(-1); mse = mean_squared_error(y, p)
    return {"MAE": float(mean_absolute_error(y, p)), "RMSE": float(np.sqrt(mse)), "MSE": float(mse), "R2": float(r2_score(y, p))}


def horizon_metrics(y, p, persistence):
    out = {}
    for h in range(y.shape[1]):
        mae = mean_absolute_error(y[:, h], p[:, h]); pmae = mean_absolute_error(y[:, h], persistence[:, h])
        out[f"MAE@{h+1}"] = float(mae); out[f"Skill@{h+1}"] = float(1 - mae / max(pmae, 1e-8))
    return out


def train_one(path, args):
    values, dates = read_hongfu(str(path)); split = chronological_split(values, dates)
    scaler = TrainMinMaxScaler().fit(split.train); z = scaler.transform(values)
    a, b = len(split.train), len(split.train) + len(split.val)
    windows = [make_decomposed_windows(z, args.look_back, args.horizon, args.n_high, s, e, args.decomp_context, args.trend_window) for s, e in ((0, a), (a, b), (b, len(z)))]
    (l_tr, h_tr, ly_tr, hy_tr, m_tr, _), (l_va, h_va, ly_va, hy_va, m_va, _), (l_te, h_te, ly_te, hy_te, m_te, _) = windows
    total_tr, total_va, total_te = ly_tr + hy_tr.sum(-1), ly_va + hy_va.sum(-1), ly_te + hy_te.sum(-1)
    base_low_tr, base_low_va, base_low_te = l_tr[:, -1], l_va[:, -1], l_te[:, -1]
    base_high_tr, base_high_va, base_high_te = h_tr[:, -1, :], h_va[:, -1, :], h_te[:, -1, :]
    base_total_tr = base_low_tr + base_high_tr.sum(-1); base_total_va = base_low_va + base_high_va.sum(-1); base_total_te = base_low_te + base_high_te.sum(-1)
    persistence = np.repeat(base_total_te[:, None], args.horizon, axis=1)
    residual = args.target == "residual"
    if residual:
        ly_tr, ly_va, ly_te = ly_tr - base_low_tr[:, None], ly_va - base_low_va[:, None], ly_te - base_low_te[:, None]
        hy_tr, hy_va, hy_te = hy_tr - base_high_tr[:, None, :], hy_va - base_high_va[:, None, :], hy_te - base_high_te[:, None, :]
        total_tr, total_va, total_te = total_tr - base_total_tr[:, None], total_va - base_total_va[:, None], total_te - base_total_te[:, None]
    device = torch.device(args.device); model = DeBiaFuseV2(args.n_high + 1, args.look_back, args.horizon).to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3); loader = DataLoader(TensorDataset(torch.from_numpy(l_tr), torch.from_numpy(h_tr), torch.from_numpy(ly_tr), torch.from_numpy(hy_tr), torch.from_numpy(m_tr), torch.from_numpy(total_tr)), batch_size=args.batch_size, shuffle=True)
    vx = [torch.from_numpy(x).to(device).float() for x in (l_va, h_va, ly_va, hy_va, m_va, total_va)]
    comp_scale = np.maximum(np.percentile(np.abs(h_tr), 75, axis=(0, 1)), 1e-3); total_scale = max(float(np.percentile(np.abs(total_tr), 75)), 1e-3)
    best, state, wait = float("inf"), None, 0; start = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for lx, hx, ly, hy, mask, ty in loader:
            opt.zero_grad(); lp, hp, tp = model(lx.to(device).float(), hx.to(device).float(), mask.to(device).float())
            loss = debiafuse_loss(lp, hp, tp, ly.to(device).float(), hy.to(device).float(), ty.to(device).float(), comp_scale, total_scale); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): _, _, vp = model(*vx[:2], vx[4]); val = float(torch.mean((vp - vx[5]) ** 2))
        if val < best: best, wait, state = val, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience: break
    model.load_state_dict(state); model.eval(); train_time = time.perf_counter() - start; start = time.perf_counter()
    with torch.no_grad(): _, _, pred = model(torch.from_numpy(l_te).to(device).float(), torch.from_numpy(h_te).to(device).float(), torch.from_numpy(m_te).to(device).float())
    pred = pred.cpu().numpy(); infer_time = time.perf_counter() - start
    if residual: pred += base_total_te[:, None]; true = scaler.inverse_transform(total_te + base_total_te[:, None]); pred_raw = scaler.inverse_transform(pred)
    else: true = scaler.inverse_transform(total_te); pred_raw = scaler.inverse_transform(pred)
    pers_raw = scaler.inverse_transform(persistence)
    r = {"dataset": path.name, "model": "DeBiaFuseV2-" + args.target, "train_windows": len(l_tr), "val_windows": len(l_va), "test_windows": len(l_te), "look_back": args.look_back, "horizon": args.horizon, "training_time": train_time, "inference_time": infer_time, "parameter_count": sum(p.numel() for p in model.parameters()), "status": "ok", **score(true, pred_raw)}
    r["Skill_MAE"] = 1 - r["MAE"] / score(true, pers_raw)["MAE"]
    r.update(horizon_metrics(true, pred_raw, pers_raw))
    return r


def main():
    root = Path(__file__).resolve().parents[1]; ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default=str(root / "data" / "Hongfu" / "deflection")); ap.add_argument("--output", default="results/debiafuse_hongfu_seed42.json"); ap.add_argument("--look-back", type=int, default=24); ap.add_argument("--horizon", type=int, default=6); ap.add_argument("--n-high", type=int, default=3); ap.add_argument("--decomp-context", type=int, default=90); ap.add_argument("--trend-window", type=int, default=30); ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--patience", type=int, default=15); ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--target", choices=["direct", "residual"], default="residual"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); args = ap.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    rows = []; quality = []
    for p in sorted(Path(args.data_dir).glob("*.xls")):
        quality.append(hongfu_quality_report(str(p))); rows.append(train_one(p, args))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rows, indent=2), encoding="utf-8"); pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False); pd.DataFrame(quality).to_csv(out.parent / "data_quality_hongfu.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__": main()
