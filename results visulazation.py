import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# 1. results 目录路径
results_dir = os.path.join(os.path.dirname(__file__), "results",)

# 2. 获取所有 npz 文件
npz_files = [
    os.path.join(results_dir, f)
    for f in os.listdir(results_dir)
    if f.endswith(".npz")
]

# 3. 根据文件名解析 horizon，例如 h1, h3, h6, h12
def get_horizon(filepath):
    filename = os.path.basename(filepath)
    m = re.search(r'h(\d+)', filename)
    return int(m.group(1)) if m else None

# 4. 加载 npz 文件
def load_npz(path):
    data = np.load(path)
    return data["labels"], data["preds"]

# 5. 自动区分 train / test 文件
test_files = [
    f for f in npz_files
    if not os.path.basename(f).startswith("train")
]

train_files = [
    f for f in npz_files
    if os.path.basename(f).startswith("train")
]

# 6. 排序（确保按照 1, 3, 6, 12 的顺序）
test_files  = sorted(test_files,  key=get_horizon)
train_files = sorted(train_files, key=get_horizon)

# 7. 测试一下加载
for f in test_files:
    h = get_horizon(f)
    labels, preds = load_npz(f)
    print(f"Loaded test horizon={h}, labels shape={labels.shape}, preds shape={preds.shape}")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---- 新增：SMAPE 函数 ----
def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE), in percentage (%).
    """
    return np.mean(
        2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

rows = []

for f in test_files:
    h = get_horizon(f)           # 例如 1, 3, 6, 12
    labels, preds = load_npz(f)  # labels, preds 形状: (N, H)

    # 展平所有样本 + 所有步长，一起算一个整体指标
    y_true = labels.reshape(-1)
    y_pred = preds.reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    smap = smape(y_true, y_pred)

    rows.append({
        "Horizon": h,
        "MSE": mse,
        "MAE": mae,
        # "RMSE": rmse,
        "SMAPE": smap,
        "R2": r2,
    })

df_metrics = pd.DataFrame(rows).sort_values("Horizon")
print(df_metrics.to_string(index=False))

# -------- 曲线可视化部分：保持不变 --------

horizons = [1, 3, 6, 12]
files_by_h = {h: next(f for f in test_files if get_horizon(f) == h) for h in horizons}

# Load labels and preds for each horizon
labels_h = {}
preds_h = {}
for h in horizons:
    labels_h[h], preds_h[h] = load_npz(files_by_h[h])

# Make sure we use the common length across all horizons
N_min = min(labels_h[h].shape[0] for h in horizons)

# --- Define the series to plot ---

# Use true(t+1) from horizon=1 as ground truth reference
y_true = labels_h[1][:N_min, 0]            # true value at t+1

# Predictions at different look-aheads
y_pred_t1  = preds_h[1][:N_min, 0]         # t+1, horizon=1
y_pred_t3  = preds_h[3][:N_min, 2]         # t+3, horizon=3 (index 2)
y_pred_t6  = preds_h[6][:N_min, 5]         # t+6, horizon=6 (index 5)
y_pred_t12 = preds_h[12][:N_min, 11]       # t+12, horizon=12 (index 11)

# Optionally only plot first 1000 samples for readability
n_show = min(1000, N_min)
x = np.arange(n_show)

plt.figure(figsize=(12, 5))
plt.plot(x, y_true[:n_show],      label="y_true (t+1)", linewidth=2)
plt.plot(x, y_pred_t1[:n_show],   label="y_pred (t+1, H=1)", linestyle="--")
plt.plot(x, y_pred_t3[:n_show],   label="y_pred (t+3, H=3)", linestyle="--")
plt.plot(x, y_pred_t6[:n_show],   label="y_pred (t+6, H=6)", linestyle="--")
plt.plot(x, y_pred_t12[:n_show],  label="y_pred (t+12, H=12)", linestyle="--")

plt.title("Comparison of predictions at different horizons")
plt.xlabel("Sample index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()