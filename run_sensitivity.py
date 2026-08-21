"""Run single-origin or full-series Chronos-2 sensitivity experiments."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None


TARGET_COLUMN = "Fluid Loss"
CONTROL_COLUMNS = {
    "inclination": ("Inclination", "inclination_input"),
    "flow": ("In Flow Rate", "in_flow_rate_input"),
    "thruster": ("Thruster Force", "thruster_force_input"),
}
PAST_MEASUREMENTS = [
    "Inclination", "In Flow Rate", "Thruster Force", "Weight on Bit",
    "Torque on Bit", "Drilling Speed", "diff_Distance", "diff_Depth",
    "Internal Pressure", "Annular Pressure",
]


def load_pipeline(checkpoint, device):
    checkpoint = os.path.abspath(os.path.normpath(checkpoint))
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    try:
        return BaseChronosPipeline.from_pretrained(
            checkpoint, device_map=device, local_files_only=True
        )
    except Exception as direct_error:
        print(f"Direct checkpoint load failed ({direct_error}); loading base model and local weights.")
    pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=device)
    safetensors_path = os.path.join(checkpoint, "model.safetensors")
    pytorch_path = os.path.join(checkpoint, "pytorch_model.bin")
    if os.path.isfile(safetensors_path) and load_safetensors is not None:
        state = load_safetensors(safetensors_path)
    elif os.path.isfile(pytorch_path):
        state = torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError("Checkpoint has neither model.safetensors nor pytorch_model.bin")
    pipeline.model.load_state_dict(state, strict=False)
    pipeline.model.eval()
    return pipeline


def prepare_frame(path, control_window, include_env=False, seed=42):
    frame = pd.read_csv(path)
    required = [TARGET_COLUMN] + PAST_MEASUREMENTS + [item[0] for item in CONTROL_COLUMNS.values()]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for measured, control in CONTROL_COLUMNS.values():
        frame[control] = frame[measured].rolling(control_window, min_periods=1).mean()
    if include_env:
        frame["env"] = np.random.default_rng(seed).random(len(frame))
    return frame


def build_origins(frame, context, horizon):
    controls = [item[1] for item in CONTROL_COLUMNS.values()]
    past_columns = controls + [column for column in PAST_MEASUREMENTS if column in frame.columns]
    if "env" in frame.columns:
        past_columns.insert(0, "env")
    inputs, futures, aligned = [], [], []
    for origin in range(context, len(frame) - horizon + 1):
        history = frame.iloc[origin - context:origin]
        future = frame.iloc[origin:origin + horizon]
        inputs.append({
            "target": history[TARGET_COLUMN].to_numpy(dtype=float),
            "past_covariates": {column: history[column].to_numpy(dtype=float) for column in past_columns},
            "future_covariates": {column: future[column].to_numpy(dtype=float) for column in controls},
        })
        futures.append(future[TARGET_COLUMN].to_numpy(dtype=float))
        aligned.append(origin + horizon - 1)
    return inputs, np.stack(futures), np.asarray(aligned)


def intervene(inputs, control_column, factor):
    modified = []
    for item in inputs:
        copy_item = {
            "target": item["target"].copy(),
            "past_covariates": {key: value.copy() for key, value in item["past_covariates"].items()},
            "future_covariates": {key: value.copy() for key, value in item["future_covariates"].items()},
        }
        copy_item["future_covariates"][control_column] *= factor
        modified.append(copy_item)
    return modified


def predict(pipeline, inputs, horizon, batch_size):
    predictions = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size]
        _, means = pipeline.predict_quantiles(
            batch, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9]
        )
        predictions.extend(mean.detach().cpu().numpy().reshape(-1) for mean in means)
    return np.stack(predictions)


def save_single(args, target, factor, truth, baseline, counterfactual, aligned_index):
    time = np.arange(1, args.horizon + 1) * args.step_seconds / 60.0
    delta = counterfactual - baseline
    frame = pd.DataFrame({
        "time_min": time, "y_pred_baseline": baseline,
        "y_pred_counterfactual": counterfactual, "delta": delta, "y_true": truth,
    })
    tag = f"{target}x{factor:g}"
    stem = f"sensitivity_c{args.context}_h{args.horizon}_t{aligned_index}_{tag}"
    write_outputs(frame, time, truth, baseline, counterfactual, args, target, factor, stem)


def save_all(args, target, factor, truth, baseline, counterfactual, aligned):
    baseline_last, counterfactual_last, truth_last = baseline[:, -1], counterfactual[:, -1], truth[:, -1]
    time = aligned * args.step_seconds / 60.0
    frame = pd.DataFrame({
        "aligned_index": aligned, "time_min": time, "y_true": truth_last,
        "y_pred_baseline": baseline_last, f"y_pred_{target}x{factor:g}": counterfactual_last,
        "delta": counterfactual_last - baseline_last,
    })
    data_tag = os.path.splitext(os.path.basename(args.data_file))[0].split("_")[0]
    stem = f"sensitivity_{data_tag}_c{args.context}_h{args.horizon}_{target}x{factor:g}"
    write_outputs(frame, time, truth_last, baseline_last, counterfactual_last, args, target, factor, stem)


def write_outputs(frame, time, truth, baseline, counterfactual, args, target, factor, stem):
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, stem + ".csv")
    figure_path = os.path.join(args.output_dir, stem + ".png")
    frame.to_csv(csv_path, index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    if not args.hide_truth:
        ax.plot(time, truth, label="True lost circulation", linestyle="--")
    ax.plot(time, baseline, label="Chronos-2 (baseline inputs)")
    ax.plot(time, counterfactual, label=f"Chronos-2 ({target} × {factor:g})")
    ax.set(xlabel="Time (min)", ylabel=r"Lost circulation ($m^3/min$)", title=f"Sensitivity: {target} × {factor:g}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    delta = counterfactual - baseline
    print(f"Saved {csv_path} and {figure_path}")
    print(f"Mean |delta|={np.mean(np.abs(delta))}; Max |delta|={np.max(np.abs(delta))}")


def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    pipeline = load_pipeline(args.checkpoint, device)
    frame = prepare_frame(args.data_file, args.control_window, args.include_env, args.seed)
    inputs, truth, aligned = build_origins(frame, args.context, args.horizon)
    if args.scope == "single":
        if not 0 <= args.sample_index < len(inputs):
            raise IndexError(f"sample-index must be between 0 and {len(inputs) - 1}")
        inputs, truth, aligned = [inputs[args.sample_index]], truth[[args.sample_index]], aligned[[args.sample_index]]
    baseline = predict(pipeline, inputs, args.horizon, args.batch_size)
    for target in args.targets:
        control_column = CONTROL_COLUMNS[target][1]
        for factor in args.factors:
            counterfactual = predict(
                pipeline, intervene(inputs, control_column, factor), args.horizon, args.batch_size
            )
            if args.scope == "single":
                save_single(args, target, factor, truth[0], baseline[0], counterfactual[0], int(aligned[0]))
            else:
                save_all(args, target, factor, truth, baseline, counterfactual, aligned)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-file", default="data/0428_model_v5.csv")
    parser.add_argument("--scope", choices=("single", "all"), default="all")
    parser.add_argument("--sample-index", type=int, default=500)
    parser.add_argument("--targets", nargs="+", choices=tuple(CONTROL_COLUMNS), default=["flow"])
    parser.add_argument("--factors", nargs="+", type=float, default=[0.1])
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--control-window", type=int, default=10)
    parser.add_argument("--step-seconds", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="sensitivity_outputs")
    parser.add_argument("--hide-truth", action="store_true")
    parser.add_argument("--include-env", action="store_true", help="Add the legacy synthetic environment covariate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
