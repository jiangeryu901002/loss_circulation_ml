import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========= 1. 基本路径与文件收集 =========

# 如果这个脚本和 results 在同一层：
#   your_project/
#       analyze_h3_by_context.py
#       results/
#           Chronos_multivariate_finetune_c3h3.npz
#           ...
results_dir = os.path.join(os.path.dirname(__file__), "results")

# 拿到所有 npz 文件
npz_files = [
    os.path.join(results_dir, f)
    for f in os.listdir(results_dir)
    if f.endswith(".npz")
]

# 区分 test 和 train 文件（假设 train 文件名以 "train_" 开头）
test_files = [
    f for f in npz_files
    if not os.path.basename(f).startswith("train_")
]

train_files = [
    f for f in npz_files
    if os.path.basename(f).startswith("train_")
]

print("Found test npz files:")
for f in test_files:
    print(" ", os.path.basename(f))


# ========= 2. 工具函数 =========

def load_npz(path):
    """从 npz 文件读取 labels 和 preds 数组"""
    data = np.load(path)
    labels = data["labels"]   # shape: (N, H)
    preds  = data["preds"]    # shape: (N, H)
    return labels, preds


def parse_c_h(filepath):
    """
    从文件名解析 context_len (cX) 和 horizon (hY)
    例如: Chronos_multivariate_finetune_c128h3.npz -> (128, 3)
    """
    fname = os.path.basename(filepath)
    m = re.search(r'c(\d+)h(\d+)', fname)
    if m:
        c = int(m.group(1))
        h = int(m.group(2))
        return c, h
    return None, None


def smape(y_true, y_pred, eps=1e-8):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE), 返回百分比值
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom)


# ========= 3. 只看 horizon = 3 的模型（不同 context_len） =========

horizon_target = 3
files_h3 = []   # list of (context_len, filepath)

for f in test_files:
    c, h = parse_c_h(f)
    if h == horizon_target:
        files_h3.append((c, f))

# 按 context_len 排序: c3, c16, c64, c128
files_h3 = sorted(files_h3, key=lambda x: x[0])

print("\nHorizon = 3 test files:")
for c, f in files_h3:
    print(f"  context_len = {c}: {os.path.basename(f)}")


# ========= 4. 计算 MSE / MAE / SMAPE / R2 =========

rows = []

for context_len, f in files_h3:
    labels, preds = load_npz(f)   # labels, preds: shape (N, 3) for h=3
    y_true = labels.reshape(-1)   # 展平成一维: N*3
    y_pred = preds.reshape(-1)

    mse   = mean_squared_error(y_true, y_pred)
    mae   = mean_absolute_error(y_true, y_pred)
    s_map = smape(y_true, y_pred)   # 单位: 百分比
    r2    = r2_score(y_true, y_pred)

    rows.append({
        "context_len": context_len,
        "horizon": horizon_target,
        "MSE": mse,
        "MAE": mae,
        "SMAPE": s_map,
        "R2": r2,
    })

df_h3 = pd.DataFrame(rows).sort_values("context_len")

print("\nMetrics for horizon = 3 (different context_len):")
print(df_h3.to_string(index=False))