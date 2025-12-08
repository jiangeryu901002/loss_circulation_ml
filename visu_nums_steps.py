import os
import numpy as np
import pandas as pd
import re

from utils import compute_metrics   # 你原来用的指标函数

# ==== 修改这里 ====
results_dir = "./results"   # <- 你的 npz 文件路径
pattern = r"steps(\d+)\.npz"  # 自动提取 num_steps = 20, 50, 100...
# ===================


def collect_results(results_dir):
    rows = []

    for fname in os.listdir(results_dir):
        if fname.endswith(".npz") and re.search(pattern, fname):
            full_path = os.path.join(results_dir, fname)

            num_steps = int(re.search(pattern, fname).group(1))
            data = np.load(full_path)

            labels = data["labels"]
            preds = data["preds"]

            metrics = compute_metrics(labels, preds)
            mse, mae, mape, smape, rmse, r2 = metrics

            rows.append({
                "num_steps": num_steps,
                "MSE": mse,
                "MAE": mae,
                "SMAPE": smape / 100.0,   # ★ 将百分比转为小数
                "R2": r2
            })

    rows = sorted(rows, key=lambda x: x["num_steps"])
    return pd.DataFrame(rows)


def make_latex_table(df: pd.DataFrame):
    latex = df.to_latex(
        index=False,
        float_format="%.4f",
        column_format="c" * len(df.columns)
    )
    return latex


def main():
    df = collect_results(results_dir)

    print("\n=== Summary Table ===")
    print(df)

    # # 保存 CSV
    # out_csv = os.path.join(results_dir, "num_steps_metrics.csv")
    # df.to_csv(out_csv, index=False)
    # print(f"\nSaved CSV to: {out_csv}")
    #
    # 保存 LaTeX
    latex = make_latex_table(df)
    out_tex = os.path.join(results_dir, "num_steps_metrics.tex")
    with open(out_tex, "w") as f:
        f.write(latex)

    print(f"Saved LaTeX table to: {out_tex}")
    print("\n=== LaTeX Table ===")
    print(latex)


if __name__ == "__main__":
    main()