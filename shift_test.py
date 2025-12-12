import numpy as np

def estimate_lag_steps(y_true, y_pred, max_lag=50):
    # 计算在 [-max_lag, +max_lag] 范围内，使相关性最大的 lag
    best_lag = 0
    best_corr = -1e9
    y_true = (y_true - y_true.mean()) / (y_true.std() + 1e-8)
    y_pred = (y_pred - y_pred.mean()) / (y_pred.std() + 1e-8)

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = y_true[-lag:]
            b = y_pred[:len(a)]
        elif lag > 0:
            a = y_true[:-lag]
            b = y_pred[lag:]
        else:
            a = y_true
            b = y_pred
        corr = (a * b).mean()
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_lag, best_corr

for h in [1, 3, 6, 12]:
    d = np.load(f"./results/Chronos_multivariate_finetune_c128h{h}.npz")
    y_true = d["labels"][:, -1]
    y_pred = d["preds"][:, -1]
    lag, corr = estimate_lag_steps(y_true, y_pred, max_lag=80)
    print(f"h={h}: best_lag_steps={lag}, corr={corr:.3f}")

    import numpy as np
    import pandas as pd

    # ===== 参数 =====
    npz_path = "results/Chronos_multivariate_finetune_c128h12.npz"
    context_len = 128
    h = 12
    step_seconds = 10

    # ===== 读取 =====
    data = np.load(npz_path)
    y_true_all = data["labels"]  # (N, h)
    y_pred_all = data["preds"]  # (N, h)

    # 只取 t+12
    y_true = y_true_all[:, h - 1]
    y_pred = y_pred_all[:, h - 1]

    # ===== 构造真实时间轴 =====
    # 第一个预测对应的真实时间点：
    # context_len + (h-1)
    time_steps = np.arange(len(y_true)) + context_len + (h - 1)
    time_seconds = time_steps * step_seconds
    time_minutes = time_seconds / 60.0

    # ===== 生成 DataFrame =====
    df = pd.DataFrame({
        "time_min": time_minutes,
        "y_true": y_true,
        "y_pred": y_pred
    })

    # ===== 保存 =====
    df.to_csv("c128h12_pred_vs_true.csv", index=False)

    print("Saved to c128h12_pred_vs_true.csv")