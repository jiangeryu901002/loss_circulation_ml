"""Run legacy Chronos zero-shot and fine-tuned forecasting experiments."""

import argparse
import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.chronos import ChronosPipeline
from datasets import TimeSeriesDataset
from utils import compute_metrics


DEFAULT_SETTINGS = {
    "univariate": [(6, 3), (6, 6), (6, 12), (12, 3), (12, 6), (12, 12)],
    "multivariate": [(32, 3), (32, 3), (32, 6), (32, 12)],
}


def parse_settings(values, data_type):
    if not values:
        return DEFAULT_SETTINGS[data_type]
    settings = []
    for value in values:
        try:
            context, horizon = value.lower().replace("x", ":").split(":")
            settings.append((int(context), int(horizon)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid setting {value!r}; use CONTEXT:HORIZON") from exc
    return settings


def evaluate(pipeline, dataset, horizon, batch_size, save_path, scaler=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    labels, predictions = [], []
    for x_context, _, _, x_future in tqdm(loader, total=len(loader)):
        with torch.no_grad():
            samples = pipeline.predict(x_context, horizon)
        predictions.append(samples.mean(dim=1, keepdim=True).cpu().numpy())
        labels.append(x_future.unsqueeze(1).cpu().numpy())
    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)
    payload = {"labels": labels, "preds": predictions}
    if scaler is not None:
        payload.update(mean=scaler.mean_, std=scaler.scale_)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez(save_path, **payload)
    metrics = compute_metrics(labels, predictions, scaler=scaler)
    result = dict(zip(("MSE", "MAE", "MAPE", "SMAPE", "log-RMSE", "r2"), metrics))
    print(result)
    return result


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    settings = parse_settings(args.settings, args.data_type)
    root_path = args.data_dir or ("./data/HouseTS" if args.data_type == "univariate" else "./data")
    checkpoint_offset = args.checkpoint_offset
    if checkpoint_offset is None:
        checkpoint_offset = 0 if args.data_type == "univariate" else 6
    scaler = None
    if not args.no_scaler:
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = [args.scaler_mean], [args.scaler_scale]

    for index, (context, horizon) in enumerate(settings):
        stem = f"{args.data_type}_c{context}h{horizon}"
        dataset_path = os.path.join(root_path, stem + "_test.pt")
        dataset = TimeSeriesDataset(series=torch.load(dataset_path, map_location="cpu"))
        print(f"Testing samples: {len(dataset)}; context={context}; horizon={horizon}")

        if args.mode in ("zeroshot", "both"):
            pipeline = ChronosPipeline.from_pretrained(
                args.base_model, device_map=device, torch_dtype=torch.bfloat16
            )
            evaluate(
                pipeline, dataset, horizon, args.batch_size,
                os.path.join(args.results_dir, f"Chronos_{args.data_type}_zeroshot_c{context}h{horizon}.npz"),
                scaler,
            )

        if args.mode in ("finetune", "both"):
            checkpoint = args.checkpoint_pattern.format(
                index=index + checkpoint_offset, context=context, horizon=horizon, data_type=args.data_type
            )
            pipeline = ChronosPipeline.from_pretrained(
                checkpoint, device_map=device, torch_dtype=torch.bfloat16
            )
            evaluate(
                pipeline, dataset, horizon, args.batch_size,
                os.path.join(args.results_dir, f"Chronos_{args.data_type}_finetune_c{context}h{horizon}.npz"),
                scaler,
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-type", choices=("univariate", "multivariate"), required=True)
    parser.add_argument("--mode", choices=("zeroshot", "finetune", "both"), default="both")
    parser.add_argument("--settings", nargs="*", help="Pairs such as 32:3 32:6 32:12")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--base-model", default="amazon/chronos-t5-base")
    parser.add_argument("--checkpoint-pattern", default="checkpoints/run-{index}/checkpoint-final")
    parser.add_argument("--checkpoint-offset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--scaler-mean", type=float, default=12.41033413)
    parser.add_argument("--scaler-scale", type=float, default=0.72660783)
    parser.add_argument("--no-scaler", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
