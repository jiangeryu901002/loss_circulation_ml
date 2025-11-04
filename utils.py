import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, root_mean_squared_log_error, r2_score

def process_batch(model, batch):
    inputs_price = batch["inputs"]  # shape [batch, context_len]
    outputs = batch["outputs"]  # shape [batch, horizon_len]
    input_padding = batch["input_padding"]
    freq = batch["freq"]
    exclude_key = set(["inputs", "outputs", "input_padding", "freq"])
    dyn_num_cov = {k:v.tolist() for k, v in batch.items() if k not in exclude_key}

    # 这里用 xreg_mode="xreg + timesfm"
    inputs = model.dataset_with_covariates(
        inputs=inputs_price,
        dynamic_numerical_covariates=dyn_num_cov,
        dynamic_categorical_covariates={},
        static_numerical_covariates={},
        static_categorical_covariates={},
        # freq=freq.tolist(),
        ridge=0.0,
        force_on_cpu=False,
        normalize_xreg_target_per_input=True
    )

    inputs = torch.from_numpy(np.stack(inputs)).float()
    # outputs = torch.from_numpy(np.stack(outputs)).float()
    # print('inputs outputs after xreg:', inputs.shape, outputs.shape, (inputs_price - inputs).abs().sum(),
    #       inputs_price.mean(), inputs_price.std(), inputs.mean(), inputs.std())

    if inputs.shape[1] < 32:
        padding = torch.zeros((inputs.shape[0], 32 - inputs.shape[1]))
        inputs = torch.concat([padding, inputs], dim=1)
        input_padding = torch.concat([torch.ones_like(padding), input_padding], dim=1)
    # print('inputs outputs after padding:', inputs.shape, outputs.shape)
    return inputs, input_padding, freq, outputs

def mm_scaler(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)
##############################
# 3) 指标
##############################
def mse(y_pred, y_true):
    return np.mean((np.array(y_pred) - np.array(y_true)) ** 2, axis=1)

def mae(y_pred, y_true):
    return np.mean(np.abs(np.array(y_pred) - np.array(y_true)), axis=1)

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

def compute_metrics(y_pred, y_true, scaler):
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1))# (n_samples, n_preds)
    y_pred, y_true = np.expm1(y_pred), np.expm1(y_true)
    ymin, ymax = min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())
    # y_pred, y_true = y_pred - ymin + 1, y_true - ymin + 1
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        log_rmse = root_mean_squared_log_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    except Exception as e:
        print(e)
        print('label', y_true)
        print('pred', y_pred)
        quit()
    return mse, mae, mape, smape, log_rmse, r2

