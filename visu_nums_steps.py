import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# 配置部分（按需修改）
# ==========================
results_dir = "./results"   # 存放 npz 结果文件的目录
context_len = 128           # 固定的 context length
horizons = [1,3,6,12]    # 需要对比的 horizons
filename_pattern = "Chronos_multivariate_finetune_c{c}h{h}.npz"

# 每个 time step = 10 seconds
STEP_SECONDS = 10.0
STEP_MINUTES = STEP_SECONDS / 60.0

# 如果你想只画前 N 个点（可选）
max_points = None  # e.g., 800


def load_last_step_series(context_len, horizon, base_dir, fname_pattern):
    """
    读取 npz 的 labels 和 preds，并取最后一个未来步 (t+horizon)。
    返回 y_true, y_pred (长度 N)。
    """
    fname = fname_pattern.format(c=context_len, h=horizon)
    fpath = os.path.join(base_dir, fname)

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File not found: {fpath}")

    data = np.load(fpath)
    labels = data["labels"]  # (N, horizon)
    preds  = data["preds"]   # (N, horizon)
    print(labels.shape, preds.shape)
    y_true = labels[:, -1]   # 真值对应 t+horizon
    y_pred = preds[:, -1]    # 预测对应 t+horizon

    return y_true, y_pred


def plot_pred_true_with_time_shift(context_len, horizons, base_dir, fname_pattern, max_points=None):
    """
    纵向大图：每个 horizon 一个子图，两条线 true vs pred。
    关键：把 (t+horizon) 的 true/pred 在 x 轴上整体右移 horizon 个 step。
    x 轴单位：minutes
    """
    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 11), sharex=True)
    if len(horizons) == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        y_true, y_pred = load_last_step_series(context_len, h, base_dir, fname_pattern)

        # 可选截断（在 shift 前截断即可）
        if max_points is not None:
            y_true = y_true[:max_points]
            y_pred = y_pred[:max_points]

        N = len(y_true)

        # 关键：时间轴右移 h 个 step，使其对应 t+h
        # 每个样本索引 i 对应的预测点应落在 (i + h) 处
        time_min = (np.arange(N) + context_len + (h - 1)) * STEP_MINUTES

        ax.plot(time_min, y_true, label="True", linewidth=1.6)
        ax.plot(time_min, y_pred, label="Predicted", linewidth=1.3, linestyle="--")

        ax.set_title(f"Horizon = {h} (t+{h})", loc="left")
        ax.set_ylabel(r"Lost circulation ($\mathrm{m}^3/\mathrm{min}$)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (min)")

    fig.suptitle(
        f"Predicted vs True Lost Circulation at Different Horizons (Context Length = {context_len})",
        fontsize=14, y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_name = f"pred_vs_true_shifted_c{context_len}.png"
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {out_name}")


if __name__ == "__main__":
    plot_pred_true_with_time_shift(
        context_len=context_len,
        horizons=horizons,
        base_dir=results_dir,
        fname_pattern=filename_pattern,
        max_points=max_points,
    )