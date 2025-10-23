# import timesfm
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))
from baselines.finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
from huggingface_hub import snapshot_download
from data_loader import load_CSM, compute_metrics
from baselines.datasets import prepare_datasets
timesfm_backend = "gpu"  # @param
from baselines.timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from baselines.timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
import numpy as np
from os import path
from typing import Optional, Tuple
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def get_model(context_len, horizon_len, load_weights: bool = False):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  repo_id = "google/timesfm-2.0-500m-pytorch"
  hparams = TimesFmHparams(
      backend=device,
      per_core_batch_size=32,
      horizon_len=horizon_len,
      num_layers=50,
      # use_positional_embedding=False,
      context_len=context_len,  # Context length can be anything up to 2048 in multiples of 32
  )
  tfm = TimesFm(hparams=hparams,
                checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id))

  model = PatchedTimeSeriesDecoder(tfm._model_config)
  if load_weights:
    checkpoint_path = path.join(snapshot_download(repo_id), "torch_model.ckpt")
    loaded_checkpoint = torch.load(checkpoint_path)#, weights_only=True)
    model.load_state_dict(loaded_checkpoint)
  return model, hparams, tfm._model_config

def plot_predictions(
    model: TimesFm,
    val_dataset: Dataset,
    save_path: Optional[str] = "predictions.png",
) -> None:
  """
    Plot model predictions against ground truth for a batch of validation data.

    Args:
      model: Trained TimesFM model
      val_dataset: Validation dataset
      save_path: Path to save the plot
    """

  model.eval()

  x_context, x_padding, freq, x_future = val_dataset[0]
  x_context = x_context.unsqueeze(0)  # Add batch dimension
  x_padding = x_padding.unsqueeze(0)
  freq = freq.unsqueeze(0)
  x_future = x_future.unsqueeze(0)

  device = next(model.parameters()).device
  x_context = x_context.to(device)
  x_padding = x_padding.to(device)
  freq = freq.to(device)
  x_future = x_future.to(device)


  context_vals = x_context[0].cpu().numpy()
  future_vals = x_future[0].cpu().numpy()
  context_len = context_vals.shape[-1]
  horizon_len = future_vals.shape[-1]
  channel_id = 13
  with torch.no_grad():
    predictions = model(x_context, x_padding.float(), freq)
    predictions_mean = predictions[:, :, :horizon_len][..., 0]  # [B, N, horizon_len]
    fluid_loss_pred = predictions_mean[:, channel_id, :]  # [B, horizon_len]
  pred_vals = fluid_loss_pred[0].cpu().numpy()
  print('plot data:', context_vals.shape, future_vals.shape, pred_vals.shape, predictions.shape, predictions_mean.shape, fluid_loss_pred.shape)
  plt.figure(figsize=(12, 6))

  plt.plot(np.arange(context_len),
           context_vals[channel_id],
           label="Historical Data",
           color="blue",
           linewidth=2)

  plt.plot(
      np.arange(context_len, context_len + horizon_len),
      future_vals[channel_id],
      label="Ground Truth",
      color="green",
      linestyle="--",
      linewidth=2,
  )

  plt.plot(np.arange(context_len, context_len + horizon_len),
           pred_vals,
           label="Prediction",
           color="red",
           linewidth=2)

  plt.xlabel("Time Step")
  plt.ylabel("Value")
  plt.title("TimesFM Predictions vs Ground Truth")
  plt.legend()
  plt.grid(True)

  if save_path:
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

  plt.close()

def single_gpu_example(model, train_dataset, val_dataset):
  """Basic example of finetuning TimesFM on stock data."""
  config = FinetuningConfig(batch_size=256,
                            num_epochs=5,
                            learning_rate=1e-4,
                            use_wandb=True,
                            freq_type=1,
                            log_every_n_steps=10,
                            val_check_interval=0.5,
                            use_quantile_loss=True)

  finetuner = TimesFMFinetuner(model, config)

  print("\nStarting finetuning...")
  results = finetuner.finetune(train_dataset=train_dataset,
                               val_dataset=val_dataset)

  print("\nFinetuning completed!")
  print(f"Training history: {len(results['history']['train_loss'])} epochs")

  plot_predictions(
      model=model,
      val_dataset=val_dataset,
      save_path="timesfm_predictions_finetuning.png",
  )

freq_type = 1
batch_size = 32
horizon_len, context_len = 12, 32
adj, train_loader, test_loader, scaler, names = load_CSM(None, normalization=None,
                                                  fold_id=2, in_len=context_len, seq_len=horizon_len+context_len,
                                                  split_ratio=0.1, BATCH_SIZE=batch_size,
                                                  mask_head=4, cuda=torch.cuda.is_available())

train_dataset, _ = prepare_datasets(
      series=train_loader[0],
      context_length=context_len,
      horizon_length=horizon_len,
      freq_type=freq_type,
      train_split=1,
  )

test_dataset, _ = prepare_datasets(
      series=test_loader[0],
      context_length=context_len,
      horizon_length=horizon_len,
      freq_type=freq_type,
      train_split=1,
  )

print(f"Created datasets:")
print(f"- Training samples: {len(train_dataset)}")
print(f"- Validation samples: {len(test_dataset)}")

model, hparams, model_config = get_model(context_len, horizon_len, load_weights=True)

print("Model loaded, evaluating model with zero-shot...")
plot_predictions(model, test_dataset, save_path="timesfm_predictions_zeroshot.png")

print("Finetuning model...")
single_gpu_example(model, train_dataset, test_dataset)



