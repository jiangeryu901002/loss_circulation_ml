from __future__ import division
from __future__ import print_function

import copy
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import numpy as np
import random
import math
import os
import tempfile
import torch
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

from utils_TSFM import load_CSM, compute_metrics, squeeze_diag, param2attn, add_noise
from modules.model_CSM_TSFM_GAT import GAT, GATConfig
from torch.optim import AdamW, RMSprop
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.integrations import INTEGRATION_TO_CALLBACK

from tsfm_public import TimeSeriesPreprocessor, TrackingCallback, count_parameters, get_datasets
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions



def zeroshot_forecast(tsp):
    zeroshot_model = get_model(
        TTM_MODEL_PATH,
        context_length=context_length,
        prediction_length=l_pred,
        prediction_channel_indices=tsp.prediction_channel_indices,
        num_input_channels=tsp.num_input_channels,
    )

    zeroshot_trainer = Trainer(
        model=zeroshot_model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=batch_size,
            use_cpu=False,
            label_names=["future_values"],
        ),
        compute_metrics=compute_metrics,
    )

    return zeroshot_trainer

def default_finetuned_forecast(tsp, train_dataset):
    finetune_forecast_model = get_model(
        TTM_MODEL_PATH,
        context_length=context_length,
        prediction_length=l_pred,
        num_input_channels=tsp.num_input_channels,
        decoder_mode="mix_channel",  # ch_mix:  set to mix_channel for mixing channels in history
        prediction_channel_indices=tsp.prediction_channel_indices,
    )
    print(
        "Number of params before freezing backbone",
        count_parameters(finetune_forecast_model),
    )

    # Freeze the backbone of the model
    for param in finetune_forecast_model.backbone.parameters():
        param.requires_grad = False

    # Count params
    print(
        "Number of params after freezing the backbone",
        count_parameters(finetune_forecast_model),
    )

    learning_rate, finetune_forecast_model = optimal_lr_finder(
        finetune_forecast_model,
        train_dataset,
        batch_size=batch_size,
        enable_prefix_tuning=False,
    )
    print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)

    print(f"Using learning rate = {learning_rate}")
    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=args.epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=4,
        report_to=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(OUT_DIR, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        label_names=["future_values"],
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    # Optimizer and scheduler
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=args.epochs,
        steps_per_epoch=math.ceil(len(train_dataset) / (batch_size)),
    )

    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
    )

    # Fine tune
    train_logs = finetune_forecast_trainer.train()

    return finetune_forecast_trainer

def gat_finetuned_forecast(tsp, train_dataset):
    gat_config = GATConfig(
        nfeat=adj.shape[0],
        nhid=32,
        num_node=adj.shape[0],
        num_time=context_length,
        num_pred=l_pred,
        n_head=4,
        dropout=0.1,
        # device=device,
        finetune=True)
    finetune_forecast_model = GAT(
        tsp=tsp,
        adj=adj,
        config=gat_config)
    learning_rate, finetune_forecast_model = optimal_lr_finder(
        finetune_forecast_model,
        train_dataset,
        batch_size=batch_size,
        enable_prefix_tuning=False,
    )
    print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)

    print(f"Using learning rate = {learning_rate}")
    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "output_data_gat2"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=args.epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=4,
        report_to=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(OUT_DIR, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        label_names=["future_values"],
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    # Optimizer and scheduler
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=args.epochs,
        steps_per_epoch=math.ceil(len(train_dataset) / (batch_size)),
    )

    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
    )

    # Fine tune
    train_logs = finetune_forecast_trainer.train()

    return finetune_forecast_trainer

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=4 * 14,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="csm",
                        help='Dataset to use.')
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                        choices=['AugNormAdj'],
                        help='Normalization method for the adjacency matrix.')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
    OUT_DIR = "./result"
    temp_dir = 'H:/DomainAdaptation/IGNN-CSM/temp'
    batch_size = 64
    l_window, offset = 13, 1
    context_length = 512
    l_pred = 12
    root = 3.0  # scale of the diffs
    # Load data
    adj, splits, data, data_diff, scaler, scaler_diff = load_CSM(args.dataset, args.normalization,
                                                                 fold_id=2,
                                                                 split_ratio=0.1, BATCH_SIZE=batch_size,
                                                                 l_window=l_window, offset=offset,
                                                                 mask_head=4, cuda=args.cuda)
    _, adj = adj[0], adj[1]  # returns the adj and mask (call the mask as adj for now)
    print('Finished data loading...')
    variables = ['inclination', 'nv_TOTAL_FLOW', 'Load', 'DH_Weigh_On_Bit',
                 'Speed*', 'Distance', 'DH_Torque', 'Internal_pressure',
                 'Q_Loss', 'Annular_Pressure']
    # dataset settings
    column_specifiers = {
        # "timestamp_column": None,
        "id_columns": [],
        "target_columns": variables,
        "observable_columns": ['env'],
        "control_columns": ['load_sma10', 'flow_sma10', 'incl_sma10'],
        # "conditional_columns": variables,
    }

    split_params = {"train": [0, splits[0]], "valid": [splits[0], splits[1]], "test": [splits[1], splits[2]]}
    # Model and optimizer
    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=l_pred,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    train_dataset, valid_dataset, test_dataset = get_datasets(tsp, data, split_params)
    train_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])
    channel_idx = tsp.prediction_channel_indices
    for k, v in train_dataset[0].items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)
    print(f'training: {len(train_dataset)}, validation: {len(valid_dataset)}, test: {len(test_dataset)}',
          f'# input channels: {tsp.num_input_channels}', 'To be predicted:', channel_idx)

    trainer = gat_finetuned_forecast(tsp, train_dataset) #default_finetuned_forecast(tsp, train_dataset) #zeroshot_forecast(tsp)
    results = trainer.evaluate(test_dataset)
    print('results:', results)

    # plot
    for c in range(len(variables)):
        plot_predictions(
            model=trainer.model,
            dset=test_dataset,
            plot_dir=os.path.join(OUT_DIR, "TSFM_data_gat2"),
            plot_prefix="test_zeroshot",
            channel=c,
            indices=[0, 5, 20, 100, 500, 700, 1300]
        )