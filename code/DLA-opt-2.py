#%% 导入核心库
import pandas as pd
import numpy as np
import os
import time
from PyEMD import EMD
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% 1. 配置参数（贴合论文实验设置）
class Config:
    def __init__(self):
        # 数据参数
        self.data_path = 'data/Hongfu/deflection/'  # 数据路径（与原代码一致）
        self.date_start = '2023-06-08'             # 画图日期起始（与原代码一致）
        self.date_end = '2023-12-15'               # 画图日期结束（与原代码一致）
        # 分解参数
        self.moving_avg_window = 30                # 移动平均窗口（月级，提取长期趋势）
        self.imf_high_threshold = 0.5              # IMF高频判断阈值（方差占比>0.5为高频）
        # 模型参数
        self.look_back = 24                        # 历史步长（短期预测L=24，长期可设为60）
        self.look_forward = 6                      # 预测步长（短期预测P=6，长期可设为30）
        self.ar_order = 12                         # AR模型阶数（用PACF图验证，默认12）
        self.lstm_units = 64                       # LSTM单元数
        self.attention_units = 32                  # 注意力层隐藏单元数
        # 训练参数
        self.batch_size = 64                       # 批次大小（论文设置）
        self.pretrain_epochs = 50                  # 子模型预训练轮数
        self.joint_epochs = 100                    # 联合微调轮数
        self.alpha = 0.4                           # 子模型损失权重（论文建议0.3-0.5）
        self.beta = 0.6                            # 整体损失权重（α+β=1）
        self.patience = 20                         # EarlyStopping耐心值

params = Config()

#%% 2. 数据读取与预处理（与原代码一致，保证兼容性）
def read_data(filepath):
    data = pd.read_excel(filepath)
    data['时间'] = data['时间'].astype(str)
    df = data.groupby(data['时间'].str[:10])['测量的桥面系挠度值'].mean().reset_index(name='mean_value')
    values = df['mean_value'].values.astype(float)
    return values, df['时间'].values

#%% 3. 数据分解（严格还原论文步骤）
def decompose_data(dataset, params):
    """
    论文DLA分解逻辑：
    1. 移动平均（AvgPool+Padding）→ 长期趋势X^T + 残差X^R
    2. EMD分解X^R → 季节IMF（X^S） + 高频扰动X^D
    3. 划分X^S为低频IMF（X^SL）和高频IMF（X^SH）
    4. 最终输出：低中频数据（X^T + X^SL）、高频数据（X^SH + X^D）
    """
    n = len(dataset)
    window = params.moving_avg_window
    
    # Step 1: 移动平均提取长期趋势X^T（带Padding，保证长度一致）
    pad_left = (window - 1) // 2
    pad_right = window // 2
    dataset_padded = np.pad(dataset, (pad_left, pad_right), mode='edge')
    X_T = np.convolve(dataset_padded, np.ones(window)/window, mode='valid')
    
    # Step 2: 计算残差X^R = 原始数据 - 长期趋势
    X_R = dataset - X_T
    
    # Step 3: EMD分解X^R为IMF（X^S）和残差（X^D，高频扰动）
    emd = EMD()
    imfs = emd(X_R)  # imfs: [IMF1, IMF2, ..., IMFk]
    X_D = emd.residue  # EMD最后的残差（高频扰动）
    
    # Step 4: 划分IMF为低频（X^SL）和高频（X^SH）（按方差占比）
    imf_vars = [np.var(imf) for imf in imfs]
    total_imf_var = sum(imf_vars)
    imfs_low, imfs_high = [], []
    for imf, var in zip(imfs, imf_vars):
        if var / total_imf_var <= params.imf_high_threshold:
            imfs_low.append(imf)
        else:
            imfs_high.append(imf)

    return X_T, imfs_low, imfs_high, X_D

#%% 4. 时间序列滑窗构造（适配AR和LSTM的输入格式）
def create_sliding_windows(dataset, look_back, look_forward):
    """
    构造滑窗数据：
    - 输入：dataset（1D数组）, look_back（历史步长）, look_forward（预测步长）
    - 输出：X（样本数, look_back）, Y（样本数, look_forward）
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back - look_forward + 1):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back:i+look_back+look_forward])
    return np.array(X), np.array(Y)

def get_windows_list(targets, look_back, look_forward, train_size):
    X_list, Y_list = [], []
    for target in targets:
        X, Y = create_sliding_windows(target, look_back, look_forward)
        X_list.append(X[:train_size])
        Y_list.append(Y[:train_size])
    return X_list, Y_list

#%% 5. AR子模型（预测低中频数据，论文设计）
class ARModel:
    def __init__(self, look_back, look_forward, ar_order):
        self.look_back = look_back
        self.look_forward = look_forward
        self.ar_order = ar_order
        self.model = None  # 存储预训练的AR模型
    
    def train(self, train_data):
        """预训练AR模型（用训练集最后ar_order个数据初始化）"""
        # 构造AR训练数据（需连续序列，不用滑窗，直接用完整训练集）
        self.model = AutoReg(train_data, lags=self.ar_order).fit()
        print(f"AR模型预训练完成（阶数={self.ar_order}，AIC={self.model.aic:.2f}）")
    
    def predict(self, X):
        Y_pred = []
        ar_params = self.model.params  # <-- 直接用 params，不要 .values
        intercept = ar_params[0]
        coefs = ar_params[1:]
        for x in X:
            current = list(x[-self.ar_order:])
            pred_steps = []
            for _ in range(self.look_forward):
                pred = intercept + np.dot(coefs, current[::-1])  # 注意AR顺序
                pred_steps.append(pred)
                current = current[1:] + [pred]
            Y_pred.append(pred_steps)
        return np.array(Y_pred)

#%% 6. 带时间注意力的LSTM子模型（预测高频数据，论文设计）
class TimeAttentionLayer(layers.Layer):
    def __init__(self, units):
        super(TimeAttentionLayer, self).__init__()
        self.W1 = layers.Dense(units, use_bias=False)
        self.W2 = layers.Dense(units, use_bias=False)
        self.v = layers.Dense(1, use_bias=False)
    
    def call(self, lstm_output):
        # lstm_output: (batch_size, look_back, lstm_units)
        query = self.W1(lstm_output)  # (batch_size, look_back, units)
        key = self.W2(lstm_output)    # (batch_size, look_back, units)
        score = self.v(tf.nn.tanh(query + key))  # (batch_size, look_back, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch_size, look_back, 1)
        # 正确的加权和方式
        context_vector = tf.reduce_sum(attention_weights * lstm_output, axis=1)  # (batch_size, lstm_units)
        return context_vector, attention_weights

def build_attention_lstm(look_back, look_forward, lstm_units, attention_units):
    inputs = layers.Input(shape=(look_back,))
    x = layers.Reshape((look_back, 1))(inputs)
    lstm_output = layers.LSTM(lstm_units, return_sequences=True)(x)
    attention_layer = TimeAttentionLayer(lstm_units)
    context_vector, attention_weights = attention_layer(lstm_output)
    outputs = layers.Dense(look_forward)(context_vector)
    # 只输出预测值
    model = Model(inputs=inputs, outputs=outputs)
    return model

#%% 7. 联合模型（整合AR和注意力LSTM，论文联合学习框架）
class DLAJointModel(Model):
    def __init__(self, ar_model, lstm_model, alpha, beta):
        super(DLAJointModel, self).__init__()
        self.ar_model = ar_model
        self.lstm_model = lstm_model
        self.alpha = alpha
        self.beta = beta
        for layer in self.lstm_model.layers[:-1]:
            layer.trainable = False

    def call(self, inputs):
        low_mid_X, high_X = inputs  # Change this line to unpack inputs correctly
        ar_pred = tf.py_function(self.ar_model.predict, [low_mid_X], tf.float32)
        ar_pred.set_shape((None, self.ar_model.look_forward))
        lstm_pred = self.lstm_model(high_X)
        total_pred = ar_pred + lstm_pred
        return [ar_pred, lstm_pred, total_pred]  # 只返回3个
    
    def compute_loss(self, y_true, y_pred, sample_weight=None, regularization_losses=None):
        ar_pred, lstm_pred, total_pred = y_pred
        ar_true, lstm_true, total_true = y_true
        
        # 计算各部分标准差（用于归一化损失）
        ar_std = tf.math.reduce_std(ar_true) + 1e-8
        lstm_std = tf.math.reduce_std(lstm_true) + 1e-8
        total_std = tf.math.reduce_std(total_true) + 1e-8
    
        # MSE损失（标准化）
        ar_loss = tf.keras.losses.MSE(ar_true, ar_pred) / (ar_std ** 2)
        lstm_loss = tf.keras.losses.MSE(lstm_true, lstm_pred) / (lstm_std ** 2)
        total_loss = tf.keras.losses.MSE(total_true, total_pred) / (total_std ** 2)
    
        # 加权总损失
        total_loss = self.alpha * (ar_loss + lstm_loss) + self.beta * total_loss

        # 加上正则化损失（如果有）
        if regularization_losses:
            total_loss += tf.add_n(regularization_losses)
        return total_loss

    def get_config(self):
        # 这里只能序列化可序列化的参数
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "beta": self.beta,
            # 注意：ar_model和lstm_model通常不能直接序列化
        })
        return config

    @classmethod
    def from_config(cls, config):
        # 你需要手动传递ar_model和lstm_model
        return cls(ar_model=None, lstm_model=None, alpha=config["alpha"], beta=config["beta"])

class DLAFullJointModel(Model):
    def __init__(self, ar_models, lstm_models, alpha, beta):
        super().__init__()
        self.ar_models = ar_models
        self.lstm_models = lstm_models
        self.alpha = alpha
        self.beta = beta

    def call(self, inputs):
        ar_inputs, lstm_inputs = inputs
        # ARModel不是Keras模型，不能直接调用，需要用predict
        ar_preds = [tf.py_function(model.predict, [inp], tf.float32) for model, inp in zip(self.ar_models, ar_inputs)]
        lstm_preds = [model(inp) for model, inp in zip(self.lstm_models, lstm_inputs)]
        total_pred = tf.add_n(ar_preds + lstm_preds)
        return ar_preds + lstm_preds + [total_pred]

#%% 8. 模型训练与预测流程
def train_dla_model(dataset, params):
    """
    完整训练流程：
    1. 划分训练集/测试集（8:2，与论文一致）
    2. 构造滑窗数据
    3. 预训练AR子模型
    4. 预训练注意力LSTM子模型
    5. 构建联合模型并微调
    """
    X_T, imfs_low, imfs_high, X_D = decompose_data(dataset, params)

    # Step 1: 划分训练集/测试集
    train_size = int(len(X_T) * 0.8)

    # Create sliding windows for AR model
    ar_targets = [X_T] + imfs_low
    ar_X_train_list, ar_Y_train_list = get_windows_list(ar_targets, params.look_back, params.look_forward, train_size)

    # Create sliding windows for LSTM model
    lstm_targets = imfs_high + [X_D]
    lstm_X_train_list, lstm_Y_train_list = get_windows_list(lstm_targets, params.look_back, params.look_forward, train_size)

    # Define ar_X_test and ar_Y_test for validation
    ar_X_test, ar_Y_test = get_windows_list(ar_targets, params.look_back, params.look_forward, len(ar_targets[0]) - train_size)

    # Define lstm_X_test and lstm_Y_test for validation
    lstm_X_test, lstm_Y_test = get_windows_list(lstm_targets, params.look_back, params.look_forward, len(lstm_targets[0]) - train_size)

    # Step 3: 预训练AR子模型
    ar_models = []
    for i, target in enumerate(ar_targets):
        ar_model = ARModel(params.look_back, params.look_forward, params.ar_order)
        ar_model.train(target[:train_size])  # 用原始一维序列训练
        ar_models.append(ar_model)

    early_stopping = EarlyStopping(monitor='val_loss', patience=params.patience, restore_best_weights=True)
    # Step 4: 预训练注意力LSTM子模型
    lstm_models = []
    for i, target in enumerate(lstm_targets):
        model = build_attention_lstm(params.look_back, params.look_forward, params.lstm_units, params.attention_units)
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss='mse')
        model.fit(
            lstm_X_train_list[i], lstm_Y_train_list[i],
            batch_size=params.batch_size,
            epochs=params.pretrain_epochs,
            validation_data=(lstm_X_test[i], lstm_Y_test[i]),
            callbacks=[early_stopping],
            verbose=1
        )
        lstm_models.append(model)
    
    # Step 5: 构建联合模型并微调
    joint_model = DLAFullJointModel(ar_models, lstm_models, params.alpha, params.beta)
    num_outputs = len(ar_models) + len(lstm_models)

    # 构造 total_Y_train_list 和 total_Y_test_list
    total_Y_train = np.sum(ar_Y_train_list + lstm_Y_train_list, axis=0)
    total_Y_test = np.sum(ar_Y_test + lstm_Y_test, axis=0)

    joint_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
        loss=['mse']*(num_outputs) + ['mse'],
        loss_weights=[params.alpha/num_outputs]*num_outputs + [params.beta]
    )
    joint_model.fit(
        [ar_X_train_list, lstm_X_train_list],
        ar_Y_train_list + lstm_Y_train_list + [total_Y_train],  # 增加total标签
        batch_size=params.batch_size,
        epochs=params.joint_epochs,
        validation_data=([ar_X_test, lstm_X_test], ar_Y_test + lstm_Y_test + [total_Y_test]),  # 增加total标签
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Step 6: 预测测试集
    ar_Y_pred_test, lstm_Y_pred_test, total_Y_pred_test = joint_model.predict([ar_X_test, lstm_X_test])  # Update prediction input
    
    return (ar_Y_pred_test, lstm_Y_pred_test, total_Y_pred_test), \
           (joint_model, ar_models, lstm_models)  # Ensure this returns the correct models

#%% 9. 评估与可视化（保留原代码功能，新增注意力可视化）
def evaluate_model(y_true, y_pred, model_name="Total"):
    """计算MAE、RMSE、MSE、R2"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Test Metrics:")
    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, MSE: {mse:.3f}, R2: {r2:.3f}")
    return mae, rmse, mse, r2

def save_log(file_name, metrics, imf_metrics=None):
    """保存评估日志到log文件夹"""
    if not os.path.exists('log'):
        os.makedirs('log')
    log_path = f'log/DLA-paper-{file_name}.txt'
    with open(log_path, 'w') as f:
        f.write(f"File: {file_name}\n")
        f.write(f"Total MAE: {metrics['total_mae']:.3f}\n")
        f.write(f"Total RMSE: {metrics['total_rmse']:.3f}\n")
        f.write(f"Total MSE: {metrics['total_mse']:.3f}\n")
        f.write(f"Total R2: {metrics['total_r2']:.3f}\n")
        f.write(f"AR MAE: {metrics['ar_mae']:.3f}\n")
        f.write(f"LSTM MAE: {metrics['lstm_mae']:.3f}\n")
    print(f"Log saved to {log_path}")

def plot_results(dates, y_true, y_pred, file_name, params):
    """绘制原始数据vs预测数据（与原代码格式一致）"""
    if not os.path.exists('graph/DLA-paper'):
        os.makedirs('graph/DLA-paper')
    fig, ax = plt.subplots(figsize=(40, 10))
    # 筛选画图日期范围内的数据（确保长度一致）
    plot_dates = pd.date_range(start=params.date_start, end=params.date_end)
    # 截取测试集数据（与dates对齐）
    test_dates = pd.to_datetime(dates[train_size + params.look_back:train_size + params.look_back + len(y_true)])
    # 绘制
    ax.plot(test_dates, y_true[:, 0], label='Origin (1st Step)', linestyle='-', linewidth=2, color='blue')  # 取预测第一步对比
    ax.plot(test_dates, y_pred[:, 0], label='Prediction (1st Step)', linestyle='-.', linewidth=4, color='red')
    # 日期格式设置
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    # 标签设置
    ax.set_xlabel('Date', fontsize=30)
    ax.set_ylabel('Deflection (m)', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(prop={'size': 25}, loc='lower left')
    # 保存
    save_path = f'graph/DLA-paper/{file_name}_pred.pdf'
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

def plot_attention(attention_weights, file_name, params):
    """绘制注意力权重（分析高频数据的关键时间步）"""
    fig, ax = plt.subplots(figsize=(20, 5))
    # 取第一个测试样本的注意力权重
    attention_sample = attention_weights[0].numpy().squeeze()
    # 绘制
    ax.bar(range(params.look_back), attention_sample, color='orange', alpha=0.7)
    ax.set_xlabel('Time Step (Look Back)', fontsize=20)
    ax.set_ylabel('Attention Weight', fontsize=20)
    ax.set_title(f'Attention Weights for High-Frequency Data (Sample 1)', fontsize=25)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # 保存
    save_path = f'graph/DLA-paper/{file_name}_attention.pdf'
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Attention plot saved to {save_path}")

#%% 10. 主函数（批量处理所有文件）
if __name__ == "__main__":
    # 获取数据文件夹下的所有文件
    filelist = os.listdir(params.data_path)
    filelist.sort()
    print(f"Found {len(filelist)} files: {filelist}")
    
    for file in filelist:
        print(f"\n=== Processing File: {file} ===")
        file_name = file[:8]  # 提取文件名前缀（与原代码一致）
        
        # Step 1: 读取数据
        dataset, dates = read_data(params.data_path + file)
        print(f"Data shape: {dataset.shape}, Date range: {dates[0]} ~ {dates[-1]}")
        
        # Step 2: 数据分解
        
        # Step 3: 训练DLA模型
        (ar_Y_pred, lstm_Y_pred, total_Y_pred), \
        (joint_model, ar_model, lstm_model) = train_dla_model(dataset, params)

        # 准备真实值（测试集）
        train_size = int(len(dataset) * 0.8)
        ar_Y_true = dataset[train_size + params.look_back:train_size + params.look_forward + params.look_back]
        lstm_Y_true = ar_Y_true  # 高频数据的真实值与低中
        total_Y_true = ar_Y_true  # 总体真实值与低中频相同（因为是加和）
        # 展平预测值
        ar_Y_pred = ar_Y_pred.reshape(-1, params.look_forward)
        lstm_Y_pred = lstm_Y_pred.reshape(-1, params.look_forward)
        total_Y_pred = total_Y_pred.reshape(-1, params.look_forward)
        # 截取与真实值长度一致的部分
        ar_Y_pred = ar_Y_pred[:len(ar_Y_true)]
        lstm_Y_pred = lstm_Y_pred[:len(lstm_Y_true)]
        total_Y_pred = total_Y_pred[:len(total_Y_true)]
        print(f"Test set size: {len(total_Y_true)} samples")
        
        # Step 4: 评估模型
        ar_mae, ar_rmse, ar_mse, ar_r2 = evaluate_model(ar_Y_true, ar_Y_pred, "AR (Low-Mid)")  # Update true values
        lstm_mae, lstm_rmse, lstm_mse, lstm_r2 = evaluate_model(lstm_Y_true, lstm_Y_pred, "LSTM (High)")  # Update true values
        total_mae, total_rmse, total_mse, total_r2 = evaluate_model(total_Y_true, total_Y_pred, "Total (DLA)")  # Ensure this is correct

        # Step 5: 保存日志
        metrics = {
            'total_mae': total_mae, 'total_rmse': total_rmse, 'total_mse': total_mse, 'total_r2': total_r2,
            'ar_mae': ar_mae, 'lstm_mae': lstm_mae
        }
        save_log(file_name, metrics)

        # Step 6: 保存模型
        if not os.path.exists('param/DLA-paper'):
            os.makedirs('param/DLA-paper')
        # 保存联合模型为 SavedModel 格式
        joint_model.save(f'param/DLA-paper/{file_name}_joint', save_format='tf')
        # LSTM子模型可用keras格式
        lstm_model.save(f'param/DLA-paper/{file_name}_lstm.keras')
        print(f"Models saved to param/DLA-paper/{file_name}_joint (SavedModel) and {file_name}_lstm.keras")
        
        # Step 7: 可视化结果
        train_size = int(len(dataset) * 0.8)
        plot_results(dates, total_Y_true, total_Y_pred, file_name, params)
        # plot_attention(attention_weights, file_name, params)
        
        print(f"=== File {file} Processed ===")