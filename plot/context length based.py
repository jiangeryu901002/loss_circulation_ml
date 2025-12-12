import matplotlib.pyplot as plt

# ---------------------------
# ---------------------------
context = [12, 64, 128, 256, 512]

# horizon = 3
MSE_h3   = [0.1000, 0.0280, 0.0255, 0.0253, 0.0249]
MAE_h3   = [0.0990, 0.0804, 0.0778, 0.0778, 0.0800]
SMAPE_h3 = [0.135,  0.138,  0.131,  0.134,  0.139]
R2_h3    = [0.547,  0.781,  0.799,  0.799,  0.790]

# horizon = 12
MSE_h12   = [0.680,  0.262,  0.0512, 0.0525, 0.0511]
MAE_h12   = [0.181,  0.156,  0.124,  0.126,  0.124 ]
SMAPE_h12 = [0.195,  0.205,  0.194,  0.195,  0.193 ]
R2_h12    = [0.206,  0.296,  0.553,  0.517,  0.556 ]

# ---------------------------
# ---------------------------
fig, axs = plt.subplots(2, 2, figsize=(11, 8))

for ax in axs.ravel():
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)

# ---- MSE ----
axs[0, 0].plot(context, MSE_h3,  marker="o", linewidth=2, label="Horizon = 3")
axs[0, 0].plot(context, MSE_h12, marker="s", linewidth=2, label="Horizon = 12")
axs[0, 0].set_title("MSE vs Context Length")
axs[0, 0].set_ylabel("MSE")
axs[0, 0].legend()

# ---- MAE ----
axs[0, 1].plot(context, MAE_h3,  marker="o", linewidth=2, label="Horizon = 3")
axs[0, 1].plot(context, MAE_h12, marker="s", linewidth=2, label="Horizon = 12")
axs[0, 1].set_title("MAE vs Context Length")
axs[0, 1].set_ylabel("MAE")
axs[0, 1].legend()

# ---- SMAPE ----
axs[1, 0].plot(context, SMAPE_h3,  marker="o", linewidth=2, label="Horizon = 3")
axs[1, 0].plot(context, SMAPE_h12, marker="s", linewidth=2, label="Horizon = 12")
axs[1, 0].set_title("SMAPE vs Context Length")
axs[1, 0].set_xlabel("Context Length (log scale)")
axs[1, 0].set_ylabel("SMAPE")
axs[1, 0].legend()

# ---- R^2 ----
axs[1, 1].plot(context, R2_h3,  marker="o", linewidth=2, label="Horizon = 3")
axs[1, 1].plot(context, R2_h12, marker="s", linewidth=2, label="Horizon = 12")
axs[1, 1].set_title("$R^2$ vs Context Length")
axs[1, 1].set_xlabel("Context Length (log scale)")
axs[1, 1].set_ylabel("$R^2$")
axs[1, 1].legend()

plt.suptitle("Model Performance Across Context Lengths (Horizon = 3 vs 12)",
             fontsize=16, y=1.02)

plt.tight_layout()
plt.savefig("context_four_metrics.png", dpi=300, bbox_inches="tight")
plt.show()