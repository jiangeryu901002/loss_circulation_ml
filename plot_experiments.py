"""Unified plotting entry point for forecasting experiments."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import compute_metrics


METRIC_ORDER = ("MSE", "MAE", "SMAPE", "R2")
METRIC_LABELS = {"MSE": "MSE", "MAE": "MAE", "SMAPE": "SMAPE", "R2": r"$R^2$"}

HORIZON_DATA = {
    "x": [1, 3, 6, 12],
    "series": {
        "": {
            "MSE": [0.0160, 0.0255, 0.0378, 0.0512],
            "MAE": [0.0617, 0.0778, 0.0990, 0.1240],
            "SMAPE": [0.115, 0.131, 0.161, 0.194],
            "R2": [0.871, 0.799, 0.704, 0.553],
        }
    },
}

CONTEXT_DATA = {
    "x": [12, 64, 128, 256, 512],
    "series": {
        "Horizon = 3": {
            "MSE": [0.1000, 0.0280, 0.0255, 0.0253, 0.0249],
            "MAE": [0.0990, 0.0804, 0.0778, 0.0778, 0.0800],
            "SMAPE": [0.135, 0.138, 0.131, 0.134, 0.139],
            "R2": [0.547, 0.781, 0.799, 0.799, 0.790],
        },
        "Horizon = 12": {
            "MSE": [0.680, 0.262, 0.0512, 0.0525, 0.0511],
            "MAE": [0.181, 0.156, 0.124, 0.126, 0.124],
            "SMAPE": [0.195, 0.205, 0.194, 0.195, 0.193],
            "R2": [0.206, 0.296, 0.553, 0.517, 0.556],
        },
    },
}

STEPS_DATA = {
    "x": [0.5, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400],
    "series": {
        "": {
            "MSE": [0.0579, 0.0573, 0.0570, 0.0566, 0.0561, 0.0553, 0.0552, 0.0551, 0.0523, 0.0512, 0.0500, 0.0490],
            "MAE": [0.131, 0.131, 0.130, 0.129, 0.129, 0.128, 0.128, 0.130, 0.127, 0.124, 0.124, 0.123],
            "SMAPE": [0.201, 0.200, 0.199, 0.199, 0.199, 0.199, 0.199, 0.201, 0.198, 0.194, 0.196, 0.194],
            "R2": [0.523, 0.522, 0.523, 0.522, 0.520, 0.518, 0.526, 0.545, 0.543, 0.553, 0.539, 0.552],
        }
    },
}

STATIC_PLOTS = {
    "horizon": (HORIZON_DATA, "Prediction Horizon", "Model Performance Across Prediction Horizons (Context Length = 128)", "horizon_four_metrics.png"),
    "context": (CONTEXT_DATA, "Context Length", "Model Performance Across Context Lengths (Horizon = 3 vs 12)", "context_four_metrics.png"),
    "steps": (STEPS_DATA, "Number of Fine-Tuning Steps", "Model Performance vs Number of Fine-Tuning Steps", "epoch_four_metrics.png"),
}


def finish_figure(fig, output, show):
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {output}")
    if show:
        plt.show()
    plt.close(fig)


def plot_static_metrics(mode, output=None, show=True):
    data, x_label, title, default_output = STATIC_PLOTS[mode]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    markers = ("o", "s", "^", "D")
    for ax, metric in zip(axes.ravel(), METRIC_ORDER):
        for marker, (label, values) in zip(markers, data["series"].items()):
            ax.plot(data["x"], values[metric], marker=marker, linewidth=2, label=label or None)
        ax.set_xscale("log")
        ax.set_title(f"{METRIC_LABELS[metric]} vs {x_label}")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        if len(data["series"]) > 1:
            ax.legend()
    axes[1, 0].set_xlabel(f"{x_label} (log scale)")
    axes[1, 1].set_xlabel(f"{x_label} (log scale)")
    fig.suptitle(title, fontsize=16, y=1.02)
    finish_figure(fig, output or default_output, show)


def load_result(results_dir, pattern, context, horizon):
    path = os.path.join(results_dir, pattern.format(c=context, h=horizon))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Result file not found: {path}")
    data = np.load(path)
    return data["labels"], data["preds"]


def print_result_metrics(results_dir, pattern, context, horizons, confidence_intervals=False):
    rows = []
    for horizon in horizons:
        labels, predictions = load_result(results_dir, pattern, context, horizon)
        metrics = compute_metrics(labels, predictions, compute_ci=confidence_intervals)
        rows.append(dict(zip(("MSE", "MAE", "MAPE", "SMAPE", "RMSE", "R2"), metrics), Horizon=horizon))
    frame = pd.DataFrame(rows).sort_values("Horizon")
    print(frame.to_string(index=False))
    return frame


def plot_horizon_comparison(results_dir, pattern, context, horizons, max_points, output, show):
    series = {h: load_result(results_dir, pattern, context, h) for h in horizons}
    print_result_metrics(results_dir, pattern, context, horizons)
    common_length = min(labels.shape[0] for labels, _ in series.values())
    n_show = min(max_points, common_length) if max_points else common_length
    reference_horizon = min(horizons)
    reference = series[reference_horizon][0][:n_show, 0]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_show)
    ax.plot(x, reference, label=f"y_true (t+{reference_horizon})", linewidth=2)
    for horizon in horizons:
        predictions = series[horizon][1][:n_show, horizon - 1]
        ax.plot(x, predictions, label=f"y_pred (t+{horizon}, H={horizon})", linestyle="--")
    ax.set(title="Comparison of predictions at different horizons", xlabel="Sample index", ylabel="Value")
    ax.legend()
    ax.grid(True)
    finish_figure(fig, output or f"horizon_predictions_c{context}.png", show)


def plot_shifted_forecasts(results_dir, pattern, context, horizons, max_points, output, show, confidence_intervals=True):
    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 11), sharex=True, squeeze=False)
    step_minutes = 10.0 / 60.0
    for ax, horizon in zip(axes[:, 0], horizons):
        labels, predictions = load_result(results_dir, pattern, context, horizon)
        metrics = compute_metrics(labels, predictions, compute_ci=confidence_intervals)
        baseline = np.repeat(predictions[:, [0]], predictions.shape[1], axis=1)
        baseline_metrics = compute_metrics(labels, baseline, compute_ci=confidence_intervals)
        print(f"Horizon {horizon}: model={metrics}; baseline={baseline_metrics}")
        y_true, y_pred = labels[:, -1], predictions[:, -1]
        if max_points:
            y_true, y_pred = y_true[:max_points], y_pred[:max_points]
        time_min = (np.arange(len(y_true)) + context + horizon - 1) * step_minutes
        ax.plot(time_min, y_true, label="True", linewidth=1.6)
        ax.plot(time_min, y_pred, label="Predicted", linewidth=1.3, linestyle="--")
        ax.set_title(f"Horizon = {horizon} (t+{horizon})", loc="left")
        ax.set_ylabel(r"Lost circulation ($\mathrm{m}^3/\mathrm{min}$)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
    axes[-1, 0].set_xlabel("Time (min)")
    fig.suptitle(f"Predicted vs True Lost Circulation at Different Horizons (Context Length = {context})", fontsize=14, y=0.98)
    finish_figure(fig, output or f"pred_vs_true_shifted_c{context}.png", show)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["horizon", "context", "steps", "comparison", "shifted"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--pattern", default="Chronos_multivariate_finetune_c{c}h{h}.npz")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6, 12])
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-ci", action="store_true", help="Disable bootstrap intervals in shifted mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    show = not args.no_show
    if args.mode in STATIC_PLOTS:
        plot_static_metrics(args.mode, args.output, show)
    elif args.mode == "comparison":
        plot_horizon_comparison(args.results_dir, args.pattern, args.context, args.horizons, args.max_points or 1000, args.output, show)
    else:
        plot_shifted_forecasts(args.results_dir, args.pattern, args.context, args.horizons, args.max_points, args.output, show, not args.no_ci)


if __name__ == "__main__":
    main()
