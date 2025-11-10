import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))
import torch
from baselines.timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from datasets import Dataset_Custom, convert_to_processed_dataset
from utils import load_CSM
import os

root_path = './data/csm'
file_names = ["0429_model_v5.csv", "0501_model_v5.csv", "0428_model_v5.csv"]
settings = [(32, 3), (32, 3), (32, 6), (32, 12)] # use smaller input lengths to get more obvious performance differences
for context_len, horizon_len in settings:
    adj, features, diffs, (scaler, scaler_diff), names = load_CSM(
        [os.path.join(root_path, f) for f in file_names], 
        fold_id=0, 
        context_len=context_len, 
        horizon_len=horizon_len, 
        train_ratio=0.8,
        mask_head=False
        )
    data, scale = features, scaler # modify this if you want to use diffs or other combinations
    train_dataset = Dataset_Custom(
        data=data['train'],
        size=[context_len, horizon_len],
        scale=scale,
        target_col='Fluid Loss',
        time_col='time_dt',
        data_type='univariate'
        )

    val_dataset = Dataset_Custom(
        data=data['val'],
        size=[context_len, horizon_len],
        scale=scale,
        target_col='Fluid Loss',
        time_col='time_dt',
        data_type='univariate'
        )

    test_dataset = Dataset_Custom(
        data=data['test'],
        size=[context_len, horizon_len],
        scale=scale,
        target_col='Fluid Loss',
        time_col='time_dt',
        data_type='univariate'
        )

    print(f"Created datasets:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    print(f"- Testing samples: {len(test_dataset)}")


    data_path = f"{root_path}/univariate_c{context_len}h{horizon_len}"
    convert_to_processed_dataset(data_path + "_train", time_series=train_dataset, data_type='univariate')
    convert_to_processed_dataset(data_path + "_val", time_series=val_dataset, data_type='univariate')
    convert_to_processed_dataset(data_path + "_test", time_series=test_dataset, data_type='univariate')

for context_len, horizon_len in settings:
    adj, features, diffs, (scaler, scaler_diff), names = load_CSM(
        [os.path.join(root_path, f) for f in file_names], 
        fold_id=0, 
        context_len=context_len, 
        horizon_len=horizon_len, 
        train_ratio=0.8,
        mask_head=False
        )
    data, scale = features, scaler # modify this if you want to use diffs or other combinations
    train_dataset = Dataset_Custom(
        data=data['train'],
        size=[context_len, horizon_len],
        scale=scale,
        target_col='Fluid Loss',
        time_col='time_dt',
        data_type='multivariate'
        )

    val_dataset = Dataset_Custom(
        data=data['val'],
        size=[context_len, horizon_len],
        scale=scale,
        target_col='Fluid Loss',
        time_col='time_dt',
        data_type='multivariate'
        )

    test_dataset = Dataset_Custom(
        data=data['test'],
        size=[context_len, horizon_len],
        scale=scale,
        target_col='Fluid Loss',
        time_col='time_dt',
        data_type='multivariate'
        )

    print(f"Created datasets:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    print(f"- Testing samples: {len(test_dataset)}")

    data_path = f"{root_path}/multivariate_c{context_len}h{horizon_len}"
    convert_to_processed_dataset(data_path + "_train", time_series=train_dataset, data_type='multivariate')
    convert_to_processed_dataset(data_path + "_val", time_series=val_dataset, data_type='multivariate')
    convert_to_processed_dataset(data_path + "_test", time_series=test_dataset, data_type='multivariate')


