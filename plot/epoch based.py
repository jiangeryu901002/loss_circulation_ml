import matplotlib.pyplot as plt

# ---------------------------
# ---------------------------
epoch_labels = ["zeroshot", 1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400]

# 用一个数值来代表 zeroshot（例如 0.5），方便做 log 轴
epoch_numeric = [0.5, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400]

MSE  = [0.0579, 0.0573, 0.0570, 0.0566, 0.0561, 0.0553, 0.0552, 0.0551, 0.0523, 0.0512, 0.0500, 0.0490]
MAE  = [0.131,  0.131,  0.130,  0.129,  0.129,  0.128,  0.128,  0.130,  0.127,  0.124,  0.124,  0.123 ]
SMAPE= [0.201,  0.200,  0.199,  0.199,  0.199,  0.199,  0.199,  0.201,  0.198,  0.194,  0.196,  0.194 ]
R2   = [0.523,  0.522,  0.523,  0.522,  0.520,  0.518,  0.526,  0.545,  0.543,  0.553,  0.539,  0.552 ]

# ---------------------------
# ---------------------------
fig, axs = plt.subplots(2, 2, figsize=(11, 8))

# 公共 x 轴设置：log scale
for ax in axs.ravel():
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)

# ---- MSE ----
axs[0, 0].plot(epoch_numeric, MSE, marker="o", linewidth=2)
axs[0, 0].set_title("MSE vs Training Steps")
axs[0, 0].set_ylabel("MSE")

# ---- MAE ----
axs[0, 1].plot(epoch_numeric, MAE, marker="o", linewidth=2)
axs[0, 1].set_title("MAE vs Training Steps")
axs[0, 1].set_ylabel("MAE")

# ---- SMAPE ----
axs[1, 0].plot(epoch_numeric, SMAPE, marker="o", linewidth=2)
axs[1, 0].set_title("SMAPE vs Training Steps")
axs[1, 0].set_xlabel("Number of Fine-Tuning Steps (log scale)")
axs[1, 0].set_ylabel("SMAPE")

# ---- R^2 ----
axs[1, 1].plot(epoch_numeric, R2, marker="o", linewidth=2)
axs[1, 1].set_title("$R^2$ vs Training Steps")
axs[1, 1].set_xlabel("Number of Fine-Tuning Steps (log scale)")
axs[1, 1].set_ylabel("$R^2$")

plt.suptitle("Model Performance vs Number of Fine-Tuning Steps", fontsize=16, y=1.02)

plt.tight_layout()
plt.savefig("epoch_four_metrics.png", dpi=300, bbox_inches="tight")
plt.show()