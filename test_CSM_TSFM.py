from __future__ import division
from __future__ import print_function

import copy
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import numpy as np
import random
import math
import os
import pandas as pd
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

def load_data():
    data_path = './data/test_case3.csv'
    df = pd.read_csv(data_path, header=0)
    df = df.fillna(0)
    df.loc[:, 'env'] = np.random.random(len(df))
    df_padding = pd.DataFrame(0, index=np.arange(502), columns=df.columns) # 2 dummy timestamp as training set
    df_padding.loc[:, 'env'] = df.iloc[1, 0] # env remains unchanged before starting
    dff = pd.concat([df_padding, df])
    return dff

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

def finetuned_forecast(tsp):
    model = GAT.from_pretrained(
        '../result/output_data_gat1/checkpoint-660',
        tsp=tsp,
        adj=adj)
    # model = model.load_state_dict(torch.load('../result/output_data_gat/checkpoint-660/model.safetensors', use_safetensors=True))#model.from_pretrained('./result/output_data_gat/checkpoint-660')

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=batch_size,
            use_cpu=False,
            label_names=["future_values"],
        ),
        compute_metrics=compute_metrics,
    )

    return trainer

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

    split_params = {"train": 0.3, "test": 0.3}
    # Model and optimizer
    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=l_pred,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )
    data = load_data()
    train_dataset, valid_dataset, test_dataset = get_datasets(tsp, data, split_params)

    print(f'training: {len(train_dataset)}, test: {len(test_dataset)}',
          f'# input channels: {tsp.num_input_channels}')

    trainer = finetuned_forecast(tsp) #default_finetuned_forecast(tsp, train_dataset) #zeroshot_forecast(tsp)
    predict = trainer.predict(test_dataset)
    print(predict.predictions[0].shape)
    np.save('./result/TSFM_test_case3', predict.predictions[0], allow_pickle=True)
    # results = trainer.evaluate(test_dataset)
    # print('results:', results)
    #
    # # plot
    # for c in range(len(variables)):
    #     plot_predictions(
    #         model=trainer.model,
    #         dset=test_dataset,
    #         plot_dir=os.path.join(OUT_DIR, "TSFM_data_gat_test_case3"),
    #         plot_prefix="test_zeroshot",
    #         channel=c,
    #         indices=[0]
    #     )