#%% 导入核心库
import os
import sys
import time
import math
import numpy as np
import pandas as pd
from typing import List, Tuple

from PyEMD import EMD
# from statsmodels.tsa.ar_model import AutoReg # 移除 AR 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 将 Crossformer 工程加入路径（使用绝对路径）
sys.path.append('./crossformer/Crossformer')
from cross_models.cross_former import Crossformer


#%% 1. 配置参数（新增 Crossformer/LSTM 超参）
class Config:
    def __init__(self):
        # 数据参数
        self.data_path = '../data/Hongfu/deflection/'  # 数据目录
        self.date_start = '2023-06-08'
        self.date_end = '2023-12-15'

        # 分解参数
        self.moving_avg_window = 30
        self.imf_high_threshold = 0.5

        # 窗口参数
        self.look_back = 24
        self.look_forward = 6

        # AR 模型 (已弃用，但保留原 AR 模型参数位置，或替换为 LSTM 参数)
        # self.ar_order = 12

        # 新增 LSTM 模型参数（针对低中频）
        self.lstm_input_size = 1 # 单变量时间序列
        self.lstm_hidden_size = 64
        self.lstm_num_layers = 2
        self.lstm_dropout = 0.1
        self.lstm_learning_rate = 1e-3
        self.lstm_epochs = 50 # 预训练 epoch

        # Crossformer 模型（针对高频）
        self.cf_in_len = 24             # 与 look_back 对齐
        self.cf_out_len = 6             # 与 look_forward 对齐
        self.cf_seg_len = 6
        self.cf_win_size = 2
        self.cf_factor = 5
        self.cf_d_model = 128
        self.cf_d_ff = 256
        self.cf_n_heads = 2
        self.cf_e_layers = 2
        self.cf_dropout = 0.1
        self.cf_learning_rate = 1e-3

        # 训练参数
        self.batch_size = 64
        self.pretrain_epochs = 50
        self.joint_epochs = 100
        self.alpha = 0.4
        self.beta = 0.6
        self.patience = 20
        self.use_gpu = torch.cuda.is_available()


params = Config()


#%% 2. 数据读取与预处理
def read_data(filepath):
    data = pd.read_excel(filepath)
    data['时间'] = data['时间'].astype(str)
    df = data.groupby(data['时间'].str[:10])['测量的桥面系挠度值'].mean().reset_index(name='mean_value')
    values = df['mean_value'].values.astype(float)
    return values, df['时间'].values


#%% 3. 数据分解（移动平均 + EMD）
def decompose_data(dataset: np.ndarray, params: Config):
    n = len(dataset)
    window = params.moving_avg_window

    pad_left = (window - 1) // 2
    pad_right = window // 2
    dataset_padded = np.pad(dataset, (pad_left, pad_right), mode='edge')
    X_T = np.convolve(dataset_padded, np.ones(window) / window, mode='valid')

    X_R = dataset - X_T

    emd = EMD()
    imfs = emd(X_R)
    X_D = emd.residue

    if imfs is None or len(imfs) == 0:
        # 无 IMF 时，全部视为残差高频
        return X_T, [], [], X_D

    imf_vars = [np.var(imf) for imf in imfs]
    total_imf_var = max(sum(imf_vars), 1e-8)
    imfs_low, imfs_high = [], []
    for imf, var in zip(imfs, imf_vars):
        if var / total_imf_var <= params.imf_high_threshold:
            imfs_low.append(imf)
        else:
            imfs_high.append(imf)

    return X_T, imfs_low, imfs_high, X_D


#%% 4. 滑窗构造
def create_sliding_windows(dataset: np.ndarray, look_back: int, look_forward: int):
    X, Y = [], []
    for i in range(len(dataset) - look_back - look_forward + 1):
        X.append(dataset[i:i + look_back])
        Y.append(dataset[i + look_back:i + look_back + look_forward])
    return np.array(X), np.array(Y)


def get_windows_list(targets: List[np.ndarray], look_back: int, look_forward: int, size: int):
    X_list, Y_list = [], []
    for target in targets:
        X, Y = create_sliding_windows(target, look_back, look_forward)
        X_list.append(X[:size])
        Y_list.append(Y[:size])
    return X_list, Y_list


#%% 5. LSTM 子模型（低中频，替换 ARModel）
class LSTMModel(nn.Module):
    def __init__(self, params: Config):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=params.lstm_input_size,
            hidden_size=params.lstm_hidden_size,
            num_layers=params.lstm_num_layers,
            batch_first=True,
            dropout=params.lstm_dropout
        )
        self.linear = nn.Linear(params.lstm_hidden_size, params.look_forward)

    def forward(self, x):
        # x shape: [batch_size, look_back, input_size]
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.linear(lstm_out[:, -1, :])
        # out shape: [batch_size, look_forward]
        return out

class LSTMModelWrapper:
    def __init__(self, params: Config, device: torch.device):
        self.device = device
        self.model = LSTMModel(params).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lstm_learning_rate)
        self.criterion = nn.MSELoss()
        self.best_state = None
        self.best_val = float('inf')
        self.patience = params.patience
        self.look_back = params.look_back
        self.look_forward = params.look_forward
        self.batch_size = params.batch_size

    def _build_loaders(self, X_train, Y_train, X_val, Y_val):
        # 形状转换到 [B, L, 1]
        X_train = X_train.reshape((-1, self.look_back, 1))
        X_val = X_val.reshape((-1, self.look_back, 1))
        train_ds = TimeSeriesTensorDataset(X_train, Y_train)
        val_ds = TimeSeriesTensorDataset(X_val, Y_val)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
        return train_loader, val_loader

    def pretrain(self, X_train, Y_train, X_val, Y_val, epochs: int):
        train_loader, val_loader = self._build_loaders(X_train, Y_train, X_val, Y_val)
        patience_counter = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx = bx.to(self.device).float()
                by = by.to(self.device).float()
                self.optimizer.zero_grad()
                pred = self.model(bx)
                loss = self.criterion(pred, by)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * bx.size(0)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(self.device).float()
                    by = by.to(self.device).float()
                    pred = self.model(bx)
                    loss = self.criterion(pred, by)
                    val_loss += loss.item() * bx.size(0)

            avg_train = train_loss / max(1, len(train_loader.dataset))
            avg_val = val_loss / max(1, len(val_loader.dataset))
            # print(f"LSTM Pretrain Epoch {epoch+1}: train={avg_train:.6f}, val={avg_val:.6f}") # 移除太多输出

            if avg_val < self.best_val:
                self.best_val = avg_val
                # 仅保存模型参数，优化器参数在联合训练中会更新
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    # print("LSTM 早停触发") # 移除太多输出
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            
        print(f"LSTM 模型预训练完成 (最佳验证集损失={self.best_val:.6f})")


    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        # 形状转换到 [B, L, 1]
        Xr = X.reshape((-1, self.look_back, 1)).astype(np.float32)
        ds = TimeSeriesTensorDataset(Xr, np.zeros((len(Xr), self.look_forward), dtype=np.float32))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(self.device).float()
                pred = self.model(bx).cpu().numpy()
                preds.append(pred)
        return np.concatenate(preds, axis=0).astype(np.float32)


#%% 6. Crossformer 高频子模型 (保持不变)
class TimeSeriesTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y


class CrossformerWrapper:
    def __init__(self, params: Config, device: torch.device):
        assert params.cf_in_len == params.look_back and params.cf_out_len == params.look_forward, \
            "Crossformer 的 in_len/out_len 必须与 look_back/look_forward 一致"
        self.device = device
        self.model = Crossformer(
            data_dim=1,
            in_len=params.cf_in_len,
            out_len=params.cf_out_len,
            seg_len=params.cf_seg_len,
            win_size=params.cf_win_size,
            factor=params.cf_factor,
            d_model=params.cf_d_model,
            d_ff=params.cf_d_ff,
            n_heads=params.cf_n_heads,
            e_layers=params.cf_e_layers,
            dropout=params.cf_dropout
        ).to(self.device).float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.cf_learning_rate)
        self.criterion = nn.MSELoss()
        self.best_state = None
        self.best_val = float('inf')
        self.patience = params.patience

    def _build_loaders(self, X_train, Y_train, X_val, Y_val, batch_size: int):
        # 形状转换到 [B, L, 1]
        X_train = X_train.reshape((-1, X_train.shape[1], 1))
        X_val = X_val.reshape((-1, X_val.shape[1], 1))
        train_ds = TimeSeriesTensorDataset(X_train, Y_train)
        val_ds = TimeSeriesTensorDataset(X_val, Y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        return train_loader, val_loader

    def pretrain(self, X_train, Y_train, X_val, Y_val, batch_size: int, epochs: int):
        train_loader, val_loader = self._build_loaders(X_train, Y_train, X_val, Y_val, batch_size)
        patience_counter = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx = bx.to(self.device).float()
                by = by.to(self.device).float()
                self.optimizer.zero_grad()
                pred = self.model(bx)  # [B, out_len, 1]
                pred = pred.squeeze(-1)  # [B, out_len]
                loss = self.criterion(pred, by)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * bx.size(0)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(self.device).float()
                    by = by.to(self.device).float()
                    pred = self.model(bx).squeeze(-1)
                    loss = self.criterion(pred, by)
                    val_loss += loss.item() * bx.size(0)

            avg_train = train_loss / max(1, len(train_loader.dataset))
            avg_val = val_loss / max(1, len(val_loader.dataset))
            # print(f"CF Pretrain Epoch {epoch+1}: train={avg_train:.6f}, val={avg_val:.6f}") # 移除太多输出

            if avg_val < self.best_val:
                self.best_val = avg_val
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    # print("CF 早停触发") # 移除太多输出
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            
        print(f"Crossformer 模型预训练完成 (最佳验证集损失={self.best_val:.6f})")


    def predict(self, X: np.ndarray, batch_size: int) -> np.ndarray:
        self.model.eval()
        X = X.reshape((-1, X.shape[1], 1)).astype(np.float32) # [B, L, 1]
        ds = TimeSeriesTensorDataset(X, np.zeros((len(X), X.shape[1]), dtype=np.float32))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(self.device).float()
                pred = self.model(bx).squeeze(-1).cpu().numpy()
                preds.append(pred)
        return np.concatenate(preds, axis=0).astype(np.float32)


#%% 7. 联合训练（LSTM 和 Crossformer 均可训练）
def joint_train_lstm_crossformer(
    lm_X_train_list: List[np.ndarray], # 低中频
    lm_Y_train_list: List[np.ndarray],
    hf_X_train_list: List[np.ndarray], # 高频
    hf_Y_train_list: List[np.ndarray],
    lm_models: List[LSTMModelWrapper],
    cf_models: List[CrossformerWrapper],
    params: Config
):
    """
    联合训练结构：
    - 批次内同时计算所有 LSTM 分量预测和所有 Crossformer 分量预测；
    - 将所有分量预测求和为 total_pred；
    - 损失为 (alpha * 平均分量MSE) + (beta * total MSE)；
      所有分量模型均参与反向。
    """
    device = torch.device('cuda' if params.use_gpu else 'cpu')
    criterion = nn.MSELoss()
    
    # 所有可训练模型和优化器
    all_models = lm_models + cf_models
    all_optimizers = [m.optimizer for m in all_models]
    all_X_train_list = lm_X_train_list + hf_X_train_list
    all_Y_train_list = lm_Y_train_list + hf_Y_train_list
    
    # 对齐样本长度
    num_samples = min(len(X) for X in all_X_train_list)
    num_comp = len(all_models)

    # 训练循环
    indices = np.arange(num_samples)
    for epoch in range(params.joint_epochs):
        np.random.shuffle(indices)
        
        # 训练模式
        for m in all_models:
            m.model.train()

        epoch_loss_total = 0.0
        for start in range(0, num_samples, params.batch_size):
            end = min(start + params.batch_size, num_samples)
            idx = indices[start:end]

            # 清零所有优化器
            for opt in all_optimizers:
                opt.zero_grad()
                
            # 分量预测与损失计算
            comp_preds = []
            per_comp_losses = []
            
            # LSTM/低中频预测
            for model_wrapper, X_np, Y_np in zip(lm_models, lm_X_train_list, lm_Y_train_list):
                bx = torch.from_numpy(X_np[idx].reshape(-1, params.look_back, 1)).to(device).float() # [B, L, 1]
                by = torch.from_numpy(Y_np[idx]).to(device).float() # [B, O]
                pred = model_wrapper.model(bx)
                loss_comp = criterion(pred, by)
                comp_preds.append(pred)
                per_comp_losses.append(loss_comp)

            # Crossformer/高频预测
            for model_wrapper, X_np, Y_np in zip(cf_models, hf_X_train_list, hf_Y_train_list):
                bx = torch.from_numpy(X_np[idx].reshape(-1, params.look_back, 1)).to(device).float() # [B, L, 1]
                by = torch.from_numpy(Y_np[idx]).to(device).float() # [B, O]
                pred = model_wrapper.model(bx).squeeze(-1)
                loss_comp = criterion(pred, by)
                comp_preds.append(pred)
                per_comp_losses.append(loss_comp)

            # 总和预测与总标签
            total_pred = torch.stack(comp_preds, dim=0).sum(dim=0)  # [B, O]
            total_true = torch.stack([torch.from_numpy(Y_np[idx]).to(device).float() for Y_np in all_Y_train_list], dim=0).sum(dim=0) # [B, O]

            # 损失函数
            mean_comp_loss = sum(per_comp_losses) / max(1, len(per_comp_losses))
            total_loss_term = criterion(total_pred, total_true)
            loss_total = params.alpha * mean_comp_loss + params.beta * total_loss_term
            
            # 反向传播与优化（所有模型）
            loss_total.backward()
            for opt in all_optimizers:
                opt.step()

            epoch_loss_total += loss_total.item() * (end - start)

        avg_epoch_loss = epoch_loss_total / max(1, num_samples)
        print(f"Joint Epoch {epoch+1}: total_loss={avg_epoch_loss:.6f}")


#%% 8. 训练与预测主流程 (修改 AR -> LSTM)
def train_debiafuse_lstm_crossformer(dataset: np.ndarray, params: Config):
    X_T, imfs_low, imfs_high, X_D = decompose_data(dataset, params)

    # 划分训练/测试
    train_size = int(len(X_T) * 0.8)

    # LSTM 目标（低中频）：趋势 + 低频IMF
    lm_targets = [X_T] + imfs_low
    lm_X_train_list, lm_Y_train_list = get_windows_list(lm_targets, params.look_back, params.look_forward, train_size)
    lm_X_test_list, lm_Y_test_list = get_windows_list(lm_targets, params.look_back, params.look_forward, len(lm_targets[0]) - train_size)

    # Crossformer 目标（高频）：高频IMF + 残差
    hf_targets = imfs_high + [X_D]
    hf_X_train_list, hf_Y_train_list = get_windows_list(hf_targets, params.look_back, params.look_forward, train_size)
    hf_X_test_list, hf_Y_test_list = get_windows_list(hf_targets, params.look_back, params.look_forward, len(hf_targets[0]) - train_size)

    device = torch.device('cuda' if params.use_gpu else 'cpu')
    
    # 预训练 LSTM（每个低中频分量一个模型）
    lm_models: List[LSTMModelWrapper] = []
    
    def split_train_val(X, Y, val_ratio=0.1):
        n = len(X)
        v = max(1, int(n * val_ratio))
        return (X[:-v], Y[:-v], X[-v:], Y[-v:]) if n > 1 else (X, Y, X, Y)
        
    for i, (Xtr, Ytr) in enumerate(zip(lm_X_train_list, lm_Y_train_list)):
        lm = LSTMModelWrapper(params, device)
        Xtr_tr, Ytr_tr, Xtr_val, Ytr_val = split_train_val(Xtr, Ytr, 0.1)
        print(f"--- Pretraining LSTM model for component {i+1} ---")
        lm.pretrain(Xtr_tr, Ytr_tr, Xtr_val, Ytr_val, params.lstm_epochs)
        lm_models.append(lm)

    # 预训练 Crossformer（每个高频分量一个模型）
    cf_models: List[CrossformerWrapper] = []
    for i, (Xtr, Ytr) in enumerate(zip(hf_X_train_list, hf_Y_train_list)):
        cf = CrossformerWrapper(params, device)
        Xtr_tr, Ytr_tr, Xtr_val, Ytr_val = split_train_val(Xtr, Ytr, 0.1)
        print(f"--- Pretraining Crossformer model for component {i+1} ---")
        cf.pretrain(Xtr_tr, Ytr_tr, Xtr_val, Ytr_val, params.batch_size, params.pretrain_epochs)
        cf_models.append(cf)

    # 联合训练（微调所有模型）
    print("--- Starting Joint Training ---")
    joint_train_lstm_crossformer(
        lm_X_train_list,
        lm_Y_train_list,
        hf_X_train_list,
        hf_Y_train_list,
        lm_models,
        cf_models,
        params
    )

    # 测试集预测
    # LSTM 预测之和
    lm_preds_test = [m.predict(Xt) for m, Xt in zip(lm_models, lm_X_test_list)]
    lm_sum_test = np.sum(lm_preds_test, axis=0).astype(np.float32)

    # Crossformer 预测之和
    cf_preds_test = [m.predict(Xt, params.batch_size) for m, Xt in zip(cf_models, hf_X_test_list)]
    cf_sum_test = np.sum(cf_preds_test, axis=0).astype(np.float32)

    total_pred_test = lm_sum_test + cf_sum_test

    # 真实总标签
    total_Y_test = np.sum(lm_Y_test_list + hf_Y_test_list, axis=0).astype(np.float32)

    return (lm_sum_test, cf_sum_test, total_pred_test), (lm_models, cf_models), (lm_Y_test_list, hf_Y_test_list, total_Y_test)


#%% 9. 评估与可视化与日志 (修改 AR -> LSTM)
def evaluate_model(y_true, y_pred, name="Total"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Test Metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, MSE={mse:.3f}, R2={r2:.3f}")
    return mae, rmse, mse, r2


def save_log(file_name, metrics):
    if not os.path.exists('log'):
        os.makedirs('log')
    # 日志文件名修改为 DeBiaFuse
    log_path = f'log/DeBiaFuse-{file_name}.txt'
    with open(log_path, 'w') as f:
        f.write(f"File: {file_name}\n")
        f.write(f"Total MAE: {metrics['total_mae']:.3f}\n")
        f.write(f"Total RMSE: {metrics['total_rmse']:.3f}\n")
        f.write(f"Total MSE: {metrics['total_mse']:.3f}\n")
        f.write(f"Total R2: {metrics['total_r2']:.3f}\n")
        f.write(f"LSTM MAE: {metrics['lm_mae']:.3f}\n") # 修改 AR -> LSTM
        f.write(f"CF MAE: {metrics['cf_mae']:.3f}\n")
    print(f"Log saved to {log_path}")


#%% 10. 主函数（批量处理）(修改 AR -> LSTM)
if __name__ == "__main__":
    filelist = os.listdir(params.data_path)
    filelist.sort()
    print(f"Found {len(filelist)} files: {filelist}")

    for file in filelist:
        print(f"\n=== Processing File: {file} ===")
        file_name = file[:8]
        dataset, dates = read_data(params.data_path + file)
        print(f"Data shape: {dataset.shape}, Date range: {dates[0]} ~ {dates[-1]}")

        (lm_sum_pred, cf_sum_pred, total_pred), (lm_models, cf_models), (lm_Y_test_list, hf_Y_test_list, total_Y_true) = \
            train_debiafuse_lstm_crossformer(dataset, params)

        # 与真实值长度对齐
        total_pred = total_pred[:len(total_Y_true)]
        lm_pred = lm_sum_pred[:len(lm_Y_test_list[0])]
        cf_pred = cf_sum_pred[:len(hf_Y_test_list[0])]

        # 评估
        # 修改 AR 为 LSTM
        lm_mae, lm_rmse, lm_mse, lm_r2 = evaluate_model(lm_Y_test_list[0], lm_pred, "LSTM (Low-Mid)")
        cf_mae, cf_rmse, cf_mse, cf_r2 = evaluate_model(hf_Y_test_list[0], cf_pred, "Crossformer (High)")
        total_mae, total_rmse, total_mse, total_r2 = evaluate_model(total_Y_true, total_pred, "Total (DeBiaFuse)")

        metrics = {
            'total_mae': total_mae, 'total_rmse': total_rmse, 'total_mse': total_mse, 'total_r2': total_r2,
            'lm_mae': lm_mae, 'cf_mae': cf_mae # 修改 ar_mae -> lm_mae
        }
        save_log(file_name, metrics)

        # 保存模型
        out_dir = 'param/DeBiaFuse' # 修改输出目录
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        # 保存 LSTM 状态
        for idx, lm in enumerate(lm_models):
            torch.save(lm.model.state_dict(), os.path.join(out_dir, f'{file_name}_lstm_{idx}.pth'))
            
        # 保存 Crossformer 状态
        for idx, cf in enumerate(cf_models):
            torch.save(cf.model.state_dict(), os.path.join(out_dir, f'{file_name}_cf_{idx}.pth'))
            
        print(f"LSTM and Crossformer models saved to {out_dir}")

        print(f"=== File {file} Processed ===")