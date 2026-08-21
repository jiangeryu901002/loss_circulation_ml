import numpy as np
import pandas as pd
import glob
import networkx as nx
import torch
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, root_mean_squared_log_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import random

random.seed(42)
np.random.seed(42)

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

def preprocess_csm(dfs, l_window):
    """
    Segment time series by sliding window -- 2*10 seconds for history, 60*10 seconds for prediction
    TODO: train on n-1 days?
    :param dfs: list of pandas dataframes
    """
    df_diffs = []
    dffs = []
    for df in dfs:
        df.loc[:, 'time_dt'] = (pd.to_datetime(df['time_dt'])-pd.Timestamp(2021, 1, 1, 0, 0, 0)).dt.total_seconds().astype(np.float32) #.apply(pd.Timestamp()//1_000_000)
        # Calculate 30-day Simple Moving Average (SMA)
        df.insert(loc=0, column='thruster_force_input', value=df['Thruster Force'].rolling(10, min_periods=1).mean())
        df.insert(loc=0, column='in_flow_rate_input', value=df['In Flow Rate'].rolling(10, min_periods=1).mean())
        df.insert(loc=0, column='inclination_input', value=df['Inclination'].rolling(10, min_periods=1).mean())
        df.insert(loc=0, column='env', value=np.random.random(len(df)))
        df_diff = df.loc[:, df.columns != 'time_dt']-df.loc[:, df.columns != 'time_dt'].shift(1)
        df_diff['time_dt'] = df['time_dt']
        df_diff.iloc[0] = df_diff.iloc[1] # consider the states before the first timestamp are all zero
        df_diff.iloc[0, 3] = 0 # env remains unchanged before starting
        df_padding = pd.DataFrame(0, index=np.arange(l_window), columns=df.columns)
        df_padding['env'] = df.iloc[1, 3] # env remains unchanged before starting
        df_diff_padding = pd.DataFrame(0, index=np.arange(l_window), columns=df.columns)
        dff = pd.concat([df_padding, df])
        df_diff = pd.concat([df_diff_padding, df_diff])
        dffs.append(dff)
        df_diffs.append(df_diff)#.map(lambda x: x**(1.0/3) if x > 0 else -(-x)**(1.0/3)))
        # df_diffs[-1] = df_diffs[-1].iloc[1:]
    scaler = std_scaler(dffs)
    scaler_diff = std_scaler(df_diffs)
    
    names = dffs[0].columns.tolist()
    print(f"number of features: {len(names)}, {names}; number of samples per day: {dffs[0].shape[0]}, {dffs[1].shape[0]}, {dffs[2].shape[0]}")
    return dffs, df_diffs, scaler, scaler_diff, names

def sliding_window(data, l_window, horizon):
    """
    Segment time series by sliding window -- l_window for history, horizon for prediction
    :param data: pandas dataframe
    :param l_window: int, length of history window
    :param horizon: int, length of prediction horizon
    :return: list of (input_window, output_window)
    """
    segments = []
    N = len(data)
    for start in range(N - l_window - horizon + 1):
        end = start + l_window
        input_window = data.iloc[start:end]
        output_window = data.iloc[end:end + horizon]
        segments.append((input_window, output_window))
    return segments

def load_CSM(file_paths, fold_id=0, context_len=11, horizon_len=1, train_ratio=0.8, 
             mask_head=False, return_dataloader=False, batch_size=32, cuda=False):
    """
    Load Citation Networks Datasets.
    """
    data = []
    for file in file_paths:
        df = pd.read_csv(file, header=0)
        df = df[['Inclination','In Flow Rate','Thruster Force','Weight on Bit','Torque on Bit',
                'Drilling Speed','diff_Distance','diff_Depth','Internal Pressure','Annular Pressure',
                'Fluid Loss', 'time_dt']]#,'Distance','Depth']]
        data.append(df)
        # print(file, df.min(), df.max())

    # three-folds validation by three days
    if fold_id == 0:
        features, diffs, scaler, scaler_diff, names = preprocess_csm(data, l_window=context_len)
    elif fold_id == 1:
        features, diffs, scaler, scaler_diff, names = preprocess_csm([data[1], data[2], data[0]], l_window=context_len)
    elif fold_id == 2:
        features, diffs, scaler, scaler_diff, names = preprocess_csm([data[2], data[0], data[1]], l_window=context_len)
    else:
        raise ValueError('Invalid fold_id')

    adj = np.zeros((14, 14))

    adj[0, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # other factors ~ N(0, 1)
    adj[1, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # inclination input
    adj[2, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # inflow rate input
    adj[3, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # thruster force (Load) input
    adj[4, :] = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # inclination
    adj[5, :] = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # inflow rate
    adj[6, :] = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # thruster force (Load)
    adj[7, :] = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])  # weight at bit
    adj[8, :] = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])  # drilling speed
    adj[9, :] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])  # hole length
    adj[10, :] = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])  # torque at bit
    adj[11, :] = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0])  # internal pressure
    adj[12, :] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])  # fluid loss
    adj[13, :] = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0])  # annular pressure
    adj += np.eye(14)
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #TODO: why do this?

    if mask_head > 0:
        As = [adj]
        for i in range(mask_head-1):
            As.append(As[-1]@adj)
        mask = np.stack(As, axis=-1)
        mask = (mask > 1e-6).astype(float)
        mask = torch.from_numpy(mask)
        if cuda:
            mask = mask.cuda()

    data = sliding_window(features[0], context_len, horizon_len) + sliding_window(features[1], context_len, horizon_len)
    random.shuffle(data)
    test_data = sliding_window(features[2], context_len, horizon_len)
    train_data = data[:int(len(data)*train_ratio)]
    val_data = data[int(len(data)*train_ratio):]

    diff = sliding_window(diffs[0], context_len, horizon_len) + sliding_window(diffs[1], context_len, horizon_len)
    random.shuffle(diff)
    test_diff = sliding_window(diffs[2], context_len, horizon_len)
    train_diff = diff[:int(len(diff)*train_ratio)]
    val_diff = diff[int(len(data)*train_ratio):]

    print(f"CSM data loaded. train/val/test samples: {len(train_data)}/{len(val_data)}/{len(test_data)}")
    print(f"input shape: {train_data[0][0].shape}, output shape: {train_data[0][1].shape}")

    if return_dataloader:
        train_loader_data = torch.utils.data.DataLoader(
            [(torch.from_numpy(d[0].iloc[:, :-1].values).float(), torch.from_numpy(d[1].iloc[:, :-1].values).float())
             for d in train_data],
            batch_size=batch_size,
            shuffle=True
        )

        val_loader_data = torch.utils.data.DataLoader(
            [(torch.from_numpy(d[0].iloc[:, :-1].values).float(), torch.from_numpy(d[1].iloc[:, :-1].values).float())
             for d in val_data],
            batch_size=batch_size,
            shuffle=False
        )

        test_loader_data = torch.utils.data.DataLoader(
            [(torch.from_numpy(d[0].iloc[:, :-1].values).float(), torch.from_numpy(d[1].iloc[:, :-1].values).float())
             for d in test_data],
            batch_size=batch_size,
            shuffle=False
        )

        train_loader_diff = torch.utils.data.DataLoader(
            [(torch.from_numpy(d[0].iloc[:, :-1].values).float(), torch.from_numpy(d[1].iloc[:, :-1].values).float())
             for d in train_diff],
            batch_size=batch_size,
            shuffle=True
        )

        val_loader_diff = torch.utils.data.DataLoader(
            [(torch.from_numpy(d[0].iloc[:, :-1].values).float(), torch.from_numpy(d[1].iloc[:, :-1].values).float())
             for d in val_diff],
            batch_size=batch_size,
            shuffle=False
        )

        test_loader_diff = torch.utils.data.DataLoader(
            [(torch.from_numpy(d[0].iloc[:, :-1].values).float(), torch.from_numpy(d[1].iloc[:, :-1].values).float())
             for d in test_diff],
            batch_size=batch_size,
            shuffle=False
        )

        return adj, {'train': train_loader_data, 'val': val_loader_data, 'test': test_loader_data}, \
               {'train': train_loader_diff, 'val': val_loader_diff, 'test': test_loader_diff}, \
               (scaler, scaler_diff), names

    return (
            adj, 
            {'train': train_data, 'val': val_data, 'test': test_data}, 
            {'train': train_diff, 'val': val_diff, 'test': test_diff}, 
            (scaler, scaler_diff), 
            names
            )

def std_scaler(dfs):
    scaler = StandardScaler()
    scaler.fit(pd.concat([d.iloc[:, :-1] for d in dfs[:-1]]))#scaler.fit(pd.concat(dfs[:-1]))
    # print('scaler', dfs[0].columns)
    # print('scaler--------------------------', scaler.mean_, scaler.scale_)
    # for df in dfs:
    #     print('df distribution', pd.concat([df.mean(axis='index'), df.std(axis='index')], axis=1))
    return scaler

def mm_scaler(dfs):
    scaler = MinMaxScaler()
    scaler.fit(pd.concat([d.iloc[:, :-1] for d in dfs[:-1]]))
    print('scaler', dfs[0].columns)
    print('scaler', scaler.min_, scaler.scale_)
    for df in dfs:
        print('df distribution', pd.concat([df.min(axis='index'), df.max(axis='index')], axis=1))
    return scaler

def rob_scaler(dfs):
    scaler = RobustScaler(quantile_range=(0, 100))
    scaler.fit(pd.concat([d.iloc[:, :-1] for d in dfs[:-1]]))
    # print('scaler', dfs[0].columns)
    # print('scaler', scaler.mean_, scaler.scale_)
    # for df in dfs:
    #     print('df distribution', pd.concat([df.mean(axis='index'), df.std(axis='index')], axis=1))
    return scaler
##############################
# 3) 指标
##############################
def mse(y_pred, y_true):
    return np.mean((np.array(y_pred) - np.array(y_true)) ** 2, axis=1)

def mae(y_pred, y_true):
    return np.mean(np.abs(np.array(y_pred) - np.array(y_true)), axis=1)

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

def compute_ci_bootstrap(y, n_rounds=500, ci=90):
    """
    Compute bootstrap confidence interval for the mean of y.
    - y: array-like, shape (n_samples,) or (n_samples, n_features)
    - n_rounds: number of bootstrap samples (default 500)
    - ci: confidence level in percent (default 90)
    Returns (lower, upper) where each is scalar for 1D input or array for 2D input (per-column).
    """
    y = np.asarray(y)
    if y.size == 0:
        raise ValueError("Empty input for compute_ci_bootstrap")
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n, d = y.shape
    al = (100.0 - ci) / 2.0
    au = 100.0 - al

    boots = np.empty((n_rounds, d), dtype=float)
    for i in range(n_rounds):
        idx = np.random.randint(0, n, size=n)
        sample = y[idx, :]
        boots[i] = sample.mean(axis=0)

    lower = np.percentile(boots, al, axis=0)
    upper = np.percentile(boots, au, axis=0)

    if lower.size == 1:
        return (float(lower[0]), float(upper[0]))
    return (lower, upper)

def compute_metrics(y_true, y_pred, scaler=None, compute_ci=False):
    """Compute aggregate forecasting metrics in a consistent order.

    Returns MSE, MAE, MAPE, sMAPE, RMSE, and R2. When ``compute_ci`` is
    enabled, each item is returned as ``(estimate, lower, upper)`` using a
    90% bootstrap confidence interval over forecast samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    if scaler is not None:
        original_shape = y_true.shape
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(original_shape)
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(original_shape)
        y_pred, y_true = np.expm1(y_pred), np.expm1(y_true)

    print("output shape:", y_pred.shape, y_true.shape)

    mse_value = mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))
    mae_value = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
    mape_value = mean_absolute_percentage_error(y_true.reshape(-1), y_pred.reshape(-1))
    smape_value = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    rmse_value = root_mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))
    r2_value = r2_score(y_true.reshape(-1), y_pred.reshape(-1))
    estimates = (mse_value, mae_value, mape_value, smape_value, rmse_value, r2_value)

    if not compute_ci:
        return estimates

    sample_axes = tuple(range(1, y_true.ndim))
    error = y_pred - y_true
    mse_samples = np.mean(error ** 2, axis=sample_axes)
    mae_samples = np.mean(np.abs(error), axis=sample_axes)
    mape_samples = np.mean(
        np.abs(error) / np.maximum(np.abs(y_true), np.finfo(float).eps),
        axis=sample_axes,
    )
    smape_samples = np.mean(
        np.abs(error) / (np.abs(y_pred) + np.abs(y_true) + 1e-6),
        axis=sample_axes,
    )
    rmse_samples = np.sqrt(mse_samples)

    metric_samples = (mse_samples, mae_samples, mape_samples, smape_samples, rmse_samples)
    intervals = [compute_ci_bootstrap(values) for values in metric_samples]

    r2_bootstrap = []
    for _ in range(500):
        indices = np.random.randint(0, y_true.shape[0], size=y_true.shape[0])
        r2_bootstrap.append(
            r2_score(y_true[indices].reshape(-1), y_pred[indices].reshape(-1))
        )
    intervals.append(tuple(np.percentile(r2_bootstrap, [5, 95])))

    return tuple(
        (float(estimate), float(bounds[0]), float(bounds[1]))
        for estimate, bounds in zip(estimates, intervals)
    )

