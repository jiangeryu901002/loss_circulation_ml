import matplotlib.pyplot as plt

# ---------------------------
# 1) 数据：Horizon 实验（context = 128）
# ---------------------------
horizon = [1, 3, 6, 12]

MSE   = [0.0160, 0.0255, 0.0378, 0.0512]
MAE   = [0.0617, 0.0778, 0.0990, 0.1240]
SMAPE = [0.115,  0.131,  0.161,  0.194 ]
R2    = [0.871,  0.799,  0.704,  0.553 ]

# ---------------------------
# 2) 四子图绘制
# ---------------------------
fig, axs = plt.subplots(2, 2, figsize=(11, 8))

for ax in axs.ravel():
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)

# ---- MSE ----
axs[0, 0].plot(horizon, MSE, marker="o", linewidth=2)
axs[0, 0].set_title("MSE vs Prediction Horizon")
axs[0, 0].set_ylabel("MSE")

# ---- MAE ----
axs[0, 1].plot(horizon, MAE, marker="o", linewidth=2)
axs[0, 1].set_title("MAE vs Prediction Horizon")
axs[0, 1].set_ylabel("MAE")

# ---- SMAPE ----
axs[1, 0].plot(horizon, SMAPE, marker="o", linewidth=2)
axs[1, 0].set_title("SMAPE vs Prediction Horizon")
axs[1, 0].set_xlabel("Prediction Horizon (log scale)")
axs[1, 0].set_ylabel("SMAPE")

# ---- R^2 ----
axs[1, 1].plot(horizon, R2, marker="o", linewidth=2)
axs[1, 1].set_title("$R^2$ vs Prediction Horizon")
axs[1, 1].set_xlabel("Prediction Horizon (log scale)")
axs[1, 1].set_ylabel("$R^2$")

plt.suptitle("Model Performance Across Prediction Horizons (Context Length = 128)",
             fontsize=16, y=1.02)

plt.tight_layout()
plt.savefig("horizon_four_metrics.png", dpi=300, bbox_inches="tight")
plt.show()