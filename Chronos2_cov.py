import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from baselines.chronos import ChronosPipeline
from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from utils import compute_metrics, load_CSM

import os

# Use only 1 GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline, Chronos2Pipeline

# Load the Chronos-2 pipeline
# GPU recommended for faster inference, but CPU is also supported
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# chatgpt preferred:
def data_processing(data, target_col, past_covariate_cols, future_covariate_cols, for_training=False):
    """
    data: list of (x, y)
      - x: 过去 context_len 个时间步（DataFrame）
      - y: 未来 horizon_len 个时间步（DataFrame）
    """
    inputs = []
    outputs = []

    for x, y in data:
        # 目标变量：过去 + 未来
        past_target = x[target_col].to_numpy().reshape(-1)      # (context_len,)
        future_target = y[target_col].to_numpy().reshape(-1)    # (horizon_len,)

        if for_training:
            target_all = np.concatenate([past_target, future_target])  # (context_len + horizon_len,)
        else:
            target_all = past_target

        # 过去协变量
        past_covariates = {}
        for p in past_covariate_cols:
            past_p = x[p].to_numpy().reshape(-1)                # (context_len,)
            future_p = y[p].to_numpy().reshape(-1)              # (horizon_len,)
            if for_training:
                past_covariates[p] = np.concatenate([past_p, future_p])  # (context_len + horizon_len,)
            else:
                past_covariates[p] = past_p                     # (context_len,)

        # 未来协变量（只在未来部分有）
        future_covariates = {
            f: y[f].to_numpy().reshape(-1) for f in future_covariate_cols   # (horizon_len,)
        }

        inputs.append(
            {
                "target": target_all,
                "past_covariates": past_covariates,
                "future_covariates": future_covariates,
            }
        )

        # outputs 只保存“未来 H 步的真值”，方便后面算指标
        outputs.append(future_target)   # shape (horizon_len,)

    # 用 stack 保证无论 H=1/3/6/12，最终都是 (N, H)
    outputs = np.stack(outputs, axis=0)

    return inputs, outputs

# def data_processing(data, target_col, past_covariate_cols, future_covariate_cols, for_training=False):
#     inputs = []
#     outputs = []
#     for x, y in data:
#         inputs.append({
#             "target": x[target_col].to_numpy().squeeze() if not for_training else np.concatenate([x[target_col].to_numpy().squeeze(), y[target_col].to_numpy().squeeze()]),
#             "past_covariates": {
#                 p: x[p].to_numpy().squeeze() if not for_training else np.concatenate([x[p].to_numpy().squeeze(), y[p].to_numpy().squeeze()]) for p in past_covariate_cols
#             },
#             "future_covariates": {
#                 f: y[f].to_numpy().squeeze() for f in future_covariate_cols
#             },
#         })
#         outputs.append(y[target_col].to_numpy().squeeze())
#     outputs = np.array(outputs)
#     return inputs, outputs

def evaluation(
        pipeline: ChronosPipeline,
        val_loader: DataLoader,
        context_len: int,
        horizon_len: int,
        batch_size: int,
        save_path: Optional[str] = "predictions.npy",
) -> None:
    # ind_target = -1 # index of the target variable in multivariate setting
    inputs, outputs = val_loader
    # for batch in tqdm(val_loader, total=len(val_loader)):
    #     x, y = batch
    #     x, y = x.permute(0, 2, 1).numpy(), y.permute(0, 2, 1).numpy()  # Change to (batch_size, num_series, seq_len)
    #     with torch.no_grad():
    #         quantiles, mean = pipeline.predict_quantiles(x, prediction_length=horizon_len, quantile_levels=[0.1, 0.5, 0.9])
    #     print("Multivariate output shapes:", quantiles[0].shape, mean[0].shape)
    #     pred_vals = mean#.cpu().numpy()
    #     future_vals = y#.cpu().numpy()
    #     # print('plot data:', context_len, horizon_len, future_vals.shape, pred_vals.shape, predictions.shape)

    #     labels.append(future_vals)
    #     preds.append(pred_vals)
    quantiles, mean = pipeline.predict_quantiles(inputs, prediction_length=horizon_len, quantile_levels=[0.1, 0.5, 0.9])
    print(len(quantiles), len(mean), quantiles[0].shape, mean[0].shape)
    preds = torch.cat(mean, dim=0).cpu().numpy()
    labels = outputs
    np.savez(save_path, labels=labels, preds=preds)
    metrics = compute_metrics(labels, preds)
    res = {'MSE': metrics[0], 'MAE': metrics[1], 'MAPE': metrics[2],
        'SMAPE': metrics[3], 'RMSE': metrics[4], 'r2': metrics[5]}
    print(res)

def finetuning(pipeline, val_loader, horizon_len, batch_size):
    # Prepare data for fine-tuning using the retail sales dataset
    inputs, outputs = val_loader
    # Fine-tune the model
    finetuned_pipeline = pipeline.fit(
        inputs=inputs,
        prediction_length=horizon_len,
        num_steps=200,  # few fine-tuning steps for a quick demo
        learning_rate=1e-5,
        batch_size=batch_size,
        logging_steps=10,
    )

    return finetuned_pipeline

if __name__ == '__main__':
    # freeze_support()
    target_col, time_col = 'Fluid Loss', 'time_dt'
    past_covariate_cols = ['env', 'inclination_input', 'in_flow_rate_input', 'thruster_force_input', 
                           'Inclination','In Flow Rate','Thruster Force','Weight on Bit','Torque on Bit',
                'Drilling Speed','diff_Distance','diff_Depth','Internal Pressure','Annular Pressure']
    future_covariate_cols = ['inclination_input', 'in_flow_rate_input', 'thruster_force_input']
    batch_size = 128
    root_path = './data'  #每次可能都要改一下
    file_names = ["0429_model_v5.csv", "0501_model_v5.csv", "0428_model_v5.csv"]
    settings = [(3,3),(512,3)]
    # use smaller input lengths to get more obvious performance differences
    #上面这一行代码，第一个数字是input length，第二个是output。之所以有四个是因为repeat了四次有四个模型

    for i, (context_len, horizon_len) in enumerate(settings):
        adj, features, diffs, (scaler, scaler_diff), names = load_CSM(
            [os.path.join(root_path, f) for f in file_names], 
            fold_id=0, 
            context_len=context_len, 
            horizon_len=horizon_len, 
            train_ratio=0.8,
            mask_head=False,
            return_dataloader=False,
            batch_size=batch_size,
            )

        data, scale = features, scaler
        print(f"Created datasets:")
        print(f"- Testing samples: {len(data['test'])}")
        print(i, context_len, horizon_len, '----------------------')

        train_inputs, train_outputs = data_processing(data['train'], target_col, past_covariate_cols, future_covariate_cols, for_training=True)
        val_inputs, val_outputs = data_processing(data['val'], target_col, past_covariate_cols, future_covariate_cols, for_training=True)
        test_inputs, test_outputs = data_processing(data['test'], target_col, past_covariate_cols, future_covariate_cols)

        # print('zeroshot...')
        # evaluation(pipeline, (test_inputs, test_outputs), context_len, horizon_len, batch_size, f'./results/Chronos_multivariate_zeroshot_c{context_len}h{horizon_len}.npz')

        print('finetune...')
        # epochs = 400
        finetuned_pipeline = pipeline.fit(
            inputs=train_inputs,
            validation_inputs=val_inputs,
            prediction_length=horizon_len,
            num_steps=200,  # few fine-tuning steps for a quick demo
            learning_rate=1e-5,
            batch_size=batch_size,
            logging_steps=10,
            finetuned_ckpt_name=f'checkpoints/Chronos_multivariate_finetune_c{context_len}h{horizon_len}/checkpoint-final',#"amazon/chronos-t5-base",
        )
        evaluation(finetuned_pipeline, (train_inputs, train_outputs), context_len, horizon_len, batch_size, f'./results/train_Chronos_multivariate_finetune_c{context_len}h{horizon_len}.npz')
        evaluation(finetuned_pipeline, (test_inputs, test_outputs), context_len, horizon_len, batch_size, f'./results/Chronos_multivariate_finetune_c{context_len}h{horizon_len}.npz')

