"""Plot Chronos-2 counterfactual sensitivity results for any control input."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_LABELS = {
    "flow": "flowrate",
    "inclination": "input inclination",
    "thruster": "input thruster",
}


def result_path(input_dir, target, factor, context, horizon):
    name = f"sensitivity_0428_c{context}_h{horizon}_{target}x{factor:g}.csv"
    return os.path.join(input_dir, name)


def find_counterfactual_column(frame, target, factor):
    exact = f"y_pred_{target}x{factor:g}"
    if exact in frame.columns:
        return exact
    prefix = f"y_pred_{target}x"
    candidates = [column for column in frame.columns if column.startswith(prefix)]
    if not candidates:
        raise ValueError(f"No counterfactual column found with prefix {prefix!r}")
    def distance(column):
        try:
            return abs(float(column[len(prefix):]) - factor)
        except ValueError:
            return float("inf")
    return min(candidates, key=distance)


def plot_sensitivity(target, factors, input_dir, context, horizon, x_min, x_max, output, show):
    fig, axes = plt.subplots(len(factors), 1, figsize=(10, 3 * len(factors)), sharex=True, squeeze=False)
    label = TARGET_LABELS[target]
    for ax, factor in zip(axes[:, 0], factors):
        path = result_path(input_dir, target, factor, context, horizon)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sensitivity result not found: {path}")
        frame = pd.read_csv(path)
        x_column = "time_min" if "time_min" in frame.columns else "aligned_index"
        x = frame[x_column].to_numpy()
        baseline = frame["y_pred_baseline"].to_numpy()
        counterfactual = frame[find_counterfactual_column(frame, target, factor)].to_numpy()
        mask = np.ones(x.shape, dtype=bool)
        if x_min is not None:
            mask &= x >= x_min
        if x_max is not None:
            mask &= x <= x_max
        ax.plot(x[mask], baseline[mask], label="Original input")
        ax.plot(x[mask], counterfactual[mask], label=f"Future {label} × {factor:g}")
        ax.set_title(f"Sensitivity test (future {label} × {factor:g})")
        ax.set_ylabel(r"Lost circulation ($m^3/min$)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
    axes[-1, 0].set_xlabel("Time (min)")
    fig.tight_layout()
    if output is None:
        output = os.path.join(input_dir, "figures", f"sensitivity_input_{target}_c{context}_h{horizon}.png")
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {output}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=sorted(TARGET_LABELS))
    parser.add_argument("--factors", type=float, nargs="+", default=[0.1, 3.0, 10.0])
    parser.add_argument("--input-dir", default="sensitivity_outputs")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    plot_sensitivity(
        args.target, args.factors, args.input_dir, args.context, args.horizon,
        args.x_min, args.x_max, args.output, not args.no_show,
    )


if __name__ == "__main__":
    main()
