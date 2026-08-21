"""Run TimesFM zero-shot or fine-tuning experiments for either data layout."""

import argparse
import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import TimeSeriesDataset
from utils import compute_metrics


DEFAULT_SETTINGS = {
    "univariate": [(6, 3), (6, 6)],
    "multivariate": [(32, 3), (32, 3), (32, 6), (32, 12)],
}


def parse_settings(values, data_type):
    if not values:
        return DEFAULT_SETTINGS[data_type]
    result = []
    for value in values:
        context, horizon = value.lower().replace("x", ":").split(":")
        result.append((int(context), int(horizon)))
    return result


def get_model(context, horizon, batch_size, repository, load_weights=False, checkpoint_path=None):
    from huggingface_hub import snapshot_download
    from baselines.timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
    from baselines.timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder

    backend = "cuda" if torch.cuda.is_available() else "cpu"
    hparams = TimesFmHparams(
        backend=backend, per_core_batch_size=batch_size, horizon_len=horizon,
        num_layers=50, context_len=context,
    )
    timesfm = TimesFm(hparams=hparams, checkpoint=TimesFmCheckpoint(huggingface_repo_id=repository))
    model = PatchedTimeSeriesDecoder(timesfm._model_config)
    if load_weights:
        checkpoint_path = checkpoint_path or os.path.join(snapshot_download(repository), "torch_model.ckpt")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model, timesfm


def evaluate(model, dataset, batch_size, save_path, data_type, scaler=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    labels, predictions = [], []
    for x_context, x_padding, freq, x_future in tqdm(loader, total=len(loader)):
        device = next(model.parameters()).device
        x_context, x_padding = x_context.to(device), x_padding.to(device)
        freq, x_future = freq.to(device), x_future.to(device)
        context = x_context.shape[-1]
        horizon = x_future.shape[-1]
        with torch.no_grad():
            output = model(x_context, x_padding.float(), freq)
            prediction = output[..., 0][:, -1, context:context + horizon]
        if data_type == "univariate":
            labels.append(x_future.unsqueeze(1).cpu().numpy())
            predictions.append(prediction.unsqueeze(1).cpu().numpy())
        else:
            labels.append(x_future.squeeze().cpu().numpy())
            predictions.append(prediction.cpu().numpy())
    labels, predictions = np.concatenate(labels), np.concatenate(predictions)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    payload = {"labels": labels, "preds": predictions}
    if scaler is not None:
        payload.update(mean=scaler.mean_, std=scaler.scale_)
    np.savez(save_path, **payload)
    metrics = compute_metrics(labels, predictions, scaler=scaler)
    result = dict(zip(("MSE", "MAE", "MAPE", "SMAPE", "RMSE", "r2"), metrics))
    print(result)
    return result


def finetune(model, timesfm, train_dataset, val_dataset, args, checkpoint_path):
    from baselines.finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner

    config = FinetuningConfig(
        batch_size=args.batch_size, num_epochs=args.epochs, learning_rate=args.learning_rate,
        use_wandb=args.use_wandb, freq_type=args.freq_type,
        log_every_n_steps=args.log_every, val_check_interval=args.val_interval,
        use_quantile_loss=True, checkpoint_save_path=checkpoint_path,
    )
    results = TimesFMFinetuner(model, timesfm, config).finetune(
        train_dataset=train_dataset, val_dataset=val_dataset
    )
    print(f"Fine-tuning completed: {len(results['history']['train_loss'])} epochs")


def load_split(root, stem, split):
    return TimeSeriesDataset(series=torch.load(os.path.join(root, f"{stem}_{split}.pt"), map_location="cpu"))


def run(args):
    settings = parse_settings(args.settings, args.data_type)
    root = args.data_dir or ("./data" if args.data_type == "univariate" else "./data/csm")
    load_weights = args.load_weights
    if load_weights is None:
        load_weights = args.data_type == "univariate"
    scaler = None
    if args.data_type == "univariate" and not args.no_scaler:
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = [args.scaler_mean], [args.scaler_scale]
    for context, horizon in settings:
        stem = f"{args.data_type}_c{context}h{horizon}"
        test_dataset = load_split(root, stem, "test")
        train_dataset = load_split(root, stem, "train") if args.mode == "finetune" else None
        val_dataset = load_split(root, stem, "val") if args.mode == "finetune" else None
        weight_path = args.weight_pattern.format(data_type=args.data_type, context=context, horizon=horizon)
        model, timesfm = get_model(
            context, horizon, args.batch_size, args.repository, load_weights, weight_path
        )
        if args.mode == "finetune":
            checkpoint_path = args.checkpoint_pattern.format(
                data_type=args.data_type, context=context, horizon=horizon
            )
            finetune(model, timesfm, train_dataset, val_dataset, args, checkpoint_path)
        output = os.path.join(
            args.results_dir, f"timesFM_{args.data_type}_{args.mode}_c{context}h{horizon}.npz"
        )
        evaluate(model, test_dataset, args.batch_size, output, args.data_type, scaler)
        if args.first_only:
            break


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-type", choices=("univariate", "multivariate"), required=True)
    parser.add_argument("--mode", choices=("zeroshot", "finetune"), default=None)
    parser.add_argument("--settings", nargs="*")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--repository", default="google/timesfm-2.0-500m-pytorch")
    parser.add_argument("--weight-pattern", default="checkpoints/timesFM_{data_type}_c{context}h{horizon}.pt")
    parser.add_argument("--checkpoint-pattern", default="checkpoints/timesFM_{data_type}_c{context}h{horizon}")
    parser.add_argument("--load-weights", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--scaler-mean", type=float, default=12.41033413)
    parser.add_argument("--scaler-scale", type=float, default=0.72660783)
    parser.add_argument("--no-scaler", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--freq-type", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--val-interval", type=float, default=0.5)
    parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--first-only", action="store_true")
    args = parser.parse_args()
    if args.mode is None:
        args.mode = "zeroshot" if args.data_type == "univariate" else "finetune"
    if args.log_every is None:
        args.log_every = 5 if args.data_type == "univariate" else 10
    return args


if __name__ == "__main__":
    run(parse_args())
