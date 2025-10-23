from os import path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class TimeSeriesDataset(Dataset):
  """Dataset for time series data compatible with TimesFM."""

  def __init__(self,
               series: List[np.ndarray],
               context_length: int,
               horizon_length: int,
               freq_type: int = 0):
    """
        Initialize dataset.

        Args:
            series: Time series data
            context_length: Number of past timesteps to use as input
            horizon_length: Number of future timesteps to predict
            freq_type: Frequency type (0, 1, or 2)
        """
    if freq_type not in [0, 1, 2]:
      raise ValueError("freq_type must be 0, 1, or 2")

    self.context_length = context_length
    self.horizon_length = horizon_length
    self.freq_type = freq_type
    if len(series[0].shape) > 2: # already sliced into segments
        self.samples = series
    else:
        self.samples = []
        for s in series:
            self._prepare_samples(s)

  def _prepare_samples(self, series) -> None:
    """Prepare sliding window samples from the time series."""
    total_length = self.context_length + self.horizon_length

    for start_idx in range(0, len(series) - total_length + 1):
      end_idx = start_idx + self.context_length
      x_context = series[start_idx:end_idx]
      x_future = series[end_idx:end_idx + self.horizon_length]
      self.samples.append((x_context, x_future))

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(
      self, index: int
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_context, x_future = self.samples[index]

    x_context = torch.tensor(x_context, dtype=torch.float32).T
    x_future = torch.tensor(x_future, dtype=torch.float32).T

    input_padding = torch.zeros_like(x_context)
    freq = torch.tensor([self.freq_type], dtype=torch.long)

    input_mask = torch.ones(self.context_length)
    # print('get data:', x_context.shape, x_future.shape)
    return x_context, input_padding, freq, x_future # x_context, x_future, input_mask #

def prepare_datasets(series: List[np.ndarray],
                     context_length: int,
                     horizon_length: int,
                     freq_type: int = 0,
                     train_split: float = 0.8) -> Tuple[Dataset, Dataset]:
  """
    Prepare training and validation datasets from time series data.

    Args:
        series: Input time series data
        context_length: Number of past timesteps to use
        horizon_length: Number of future timesteps to predict
        freq_type: Frequency type (0, 1, or 2)
        train_split: Fraction of data to use for training

    Returns:
        Tuple of (train_dataset, val_dataset)
    """

  # Create datasets with specified frequency type
  train_dataset = TimeSeriesDataset(series,
                                    context_length=context_length,
                                    horizon_length=horizon_length,
                                    freq_type=freq_type)

  if train_split < 1:
      train_size = int(len(train_dataset.samples) * train_split)
      train_dataset.samples = train_dataset.samples[:train_size]
      val_data = train_dataset.samples[train_size:]

      val_dataset = TimeSeriesDataset(val_data,
                                      context_length=context_length,
                                      horizon_length=horizon_length,
                                      freq_type=freq_type)
  else:
      val_dataset = None

  return train_dataset, val_dataset


# Data pipelining with covariates
def get_batched_data_fn(
        df: pd.DataFrame,
        batch_size: int = 128,
        context_len: int = 120,
        horizon_len: int = 24,
):
    examples = defaultdict(list)

    num_examples = 0
    for country in ("FR", "BE"):
        sub_df = df[df["unique_id"] == country]
        for start in range(0, len(sub_df) - (context_len + horizon_len), horizon_len):
            num_examples += 1
            examples["country"].append(country)
            examples["inputs"].append(sub_df["y"][start:(context_end := start + context_len)].tolist())
            examples["gen_forecast"].append(sub_df["gen_forecast"][start:context_end + horizon_len].tolist())
            examples["week_day"].append(sub_df["week_day"][start:context_end + horizon_len].tolist())
            examples["outputs"].append(sub_df["y"][context_end:(context_end + horizon_len)].tolist())

    def data_fn():
        for i in range(1 + (num_examples - 1) // batch_size):
            yield {k: v[(i * batch_size): ((i + 1) * batch_size)] for k, v in examples.items()}

    return data_fn