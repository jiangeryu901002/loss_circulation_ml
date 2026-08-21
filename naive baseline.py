import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

horizons = [1, 3, 6, 12]

print("Naive baseline (npz-based approximation):\n")

for h in horizons:
    d = np.load(f"./results/Chronos_multivariate_finetune_c128h{h}.npz")
    y_true = d["labels"][:, -1]  # y(t+h)

    # naive: shift by h
    y_naive = y_true[:-h]
    y_true_cut = y_true[h:]

    mae = mean_absolute_error(y_true_cut, y_naive)
    mse = mean_squared_error(y_true_cut, y_naive)
    r2  = r2_score(y_true_cut, y_naive)

    print(f"h = {h:2d} | MAE = {mae:.4f} | MSE = {mse:.4f} | R2 = {r2:.4f}")
