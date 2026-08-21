import os.path
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict
try:
    from .dataset.arrow import ArrowWriter
except:
    pass
from utils import process_batch

class TimeSeriesDataset(Dataset):
    """Dataset for time series data compatible with TimesFM."""

    def __init__(self,
                series: Dict[str, torch.Tensor],
                target='price',
            ):
        """
        Initialize dataset.

        Args:
            series: Time series data
            context_length: Number of past timesteps to use as input
            horizon_length: Number of future timesteps to predict
            freq_type: Frequency type (0, 1, or 2)
        """
        # print(type(series['x_context']), type(series['x_context'][0]))
        self.x_context = torch.stack(series['x_context'], axis=0)
        self.x_future = torch.stack(series['x_future'], axis=0)
        self.x_padding = torch.stack(series['x_padding'], axis=0)
        self.freq = torch.stack(series['freq'], axis=0)

    def __len__(self) -> int:
        return self.x_context.shape[0]

    def __getitem__(self, index: int):
        x_context, input_padding, freq, x_future = self.x_context[index], self.x_padding[index], self.freq[index], self.x_future[index]
        return x_context, input_padding, freq, x_future  # x_context, x_future, input_mask #



class Dataset_Custom(Dataset):

    def __init__(
        self,
        data,         # 'train', 'val', or 'test'
        size=None,            # [seq_len, pred_len]
        data_type = 'multivariate',        # 多变量 -> 单输出
        target_col='Fluid Loss',       # 目标列
        time_col='time_dt',
        scale=None,
    ):
        super().__init__()

        # 1) 滑窗相关
        if size is None:
            self.context_len = 6
            self.horizon_len = 6
        else:
            self.context_len, self.horizon_len = size

        self.data_type = data_type
        self.target_col = target_col
        self.time_col = time_col
        self.scale = scale

        self.data = data

        # 存放每个zipcode的数据块
        self.series_list = []   # 普通特征 X
        self.out_list = []      # 目标列  Y
        self.xmark_list = []    # 存放 year, month 等时间戳特征

        self.__read_data__()

    def __read_data__(self):
        for x, y in self.data:
            # X / Y / X_mark
            X = x.loc[ : , x.columns != self.time_col]  # shape [n_sub, d_in], d_in=34
            if self.scale:
                X = pd.DataFrame(self.scale.transform(X), columns=X.columns)
            Y = y[self.target_col].values
            if self.data_type == 'multivariate':
                Y = Y.reshape(-1, 1) #np.log1p(y[self.target_col].values.reshape(-1, 1))  # shape [n_sub, 1]
            X_mark = pd.concat([x, y])[self.time_col].values      # shape [n_sub, 2]
            X_mark = np.array(X_mark, dtype=np.float32)
            self.series_list.append(X)
            self.out_list.append(Y)
            self.xmark_list.append(X_mark)

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, idx):
        X = self.series_list[idx]      # shape [N, 34]
        Y = self.out_list[idx]         # shape [N, 1]
        X_mark = self.xmark_list[idx]  # shape [N, 2]
        # encoder input
        seq_x = torch.from_numpy(X[self.target_col].values).float()           # [seq_len, 34]
        
        if seq_x.shape[0] < 32:
            padding = torch.zeros((32 - seq_x.shape[0], *seq_x.shape[1:]))
            seq_x = torch.concat([padding, seq_x])
            input_padding = torch.concat([torch.ones_like(padding), input_padding])
        else:
            input_padding = torch.zeros_like(seq_x)

        seq_y = torch.from_numpy(Y).float()
        freq = torch.tensor([0], dtype=torch.long)
        seq_x_mark = X_mark[:self.context_len] # [seq_len, 2]
        seq_y_mark = X_mark[self.context_len:]   # [pred_len, 2]
        seq_x_mark = torch.from_numpy(seq_x_mark).float()
        seq_y_mark = torch.from_numpy(seq_y_mark).float()

        # print(X.columns, X.shape, Y.shape, seq_x.shape, seq_y.shape, input_padding.shape, freq.shape, seq_x_mark.shape, seq_y_mark.shape)
        covs = {c:torch.from_numpy(X[c].values).float() for c in X.columns if c != self.target_col}
        covs['x_context'] = seq_x
        covs['x_future'] = seq_y
        covs['x_padding'] = input_padding
        covs['freq'] = freq
        return covs #seq_x, seq_y, seq_x_mark, seq_y_mark, covs

##############################
# 2) 数据管道
##############################
def get_batched_data_fn(
    csv_path="HouseTS.csv",
    batch_size=128,
    context_len=12,   # 改小: 12
    horizon_len=6     # 改小:  6
):


    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date")

    # 准备: 找到所有数值列 & 分别剔除 'price'
    numeric_cols = []
    categorical_cols = []
    ignore_cols = {"date", "price", "city_full"}

    for col in df.columns:
        if col in ignore_cols:
            continue
        elif col == "city" or col == "zipcode":
            categorical_cols.append(col)
        elif col == "price":
            pass
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)

    examples = defaultdict(list)
    grouped = df.groupby("zipcode")

    num_examples = 0
    for zipc, sub_df in grouped:
        sub_df = sub_df.sort_values("date").reset_index(drop=True)
        length = len(sub_df)
        prices = sub_df["price"].values

        # 假设同一zipcode下 city 不变
        if len(sub_df["city"].unique())==1:
            city_value = sub_df["city"].iloc[0]
        else:
            city_value = sub_df["city"].iloc[0]

        # 按 horizon_len 步进 => range(0, length - (context_len + horizon_len), horizon_len)
        # context_len=12 + horizon_len=6 => total 18
        # 也可改: for start in range(length - (context_len + horizon_len) + 1):
        # 视需要决定步进
        start_ = 0
        while start_ + (context_len + horizon_len) <= length:
            c_end = start_ + context_len
            inputs_price = prices[start_:c_end]
            outputs_price = prices[c_end:c_end + horizon_len]

            # 动态 numeric
            dynamic_num = {}
            for ncol in numeric_cols:
                arr = sub_df[ncol].iloc[start_: (c_end + horizon_len)].values
                if len(arr) < context_len + horizon_len:
                    continue
                dynamic_num.setdefault(ncol, arr.tolist())

            dynamic_cat = {}
            static_num = {}
            static_cat = {
                "city": city_value,
                "zipcode": str(zipc)
            }

            examples["inputs"].append(inputs_price.tolist())
            examples["outputs"].append(outputs_price.tolist())
            examples["dynamic_num"].append(dynamic_num)
            examples["dynamic_cat"].append(dynamic_cat)
            examples["static_num"].append(static_num)
            examples["static_cat"].append(static_cat)

            num_examples += 1
            start_ += horizon_len

    def data_fn():
        total_batches = 1 + (num_examples -1)//batch_size
        for i in range(total_batches):
            yield {
                k: v[i*batch_size:(i+1)*batch_size]
                for k,v in examples.items()
            }

    return data_fn

def convert_to_processed_dataset(
    path: Union[str, Path],
    time_series: Dataset_Custom,
    data_type: str,
    compression: str = "lz4",
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """

    # Set an arbitrary start time
    start = np.datetime64("2000-01-01 00:00", "s")
    dataset_tensor = {}
    for k in time_series[0].keys():
        dataset_tensor[k] = [] 
    for item in time_series:
        if data_type == "univariate":
            x_context, x_padding, freq, x_future = item['x_context'], item['x_padding'], item['freq'], item['x_future']
            # print(x_context.shape, x_padding.shape, freq.shape, x_future.shape)
            ts = torch.cat((x_context, x_future), dim=-1).numpy()
            dataset_tensor['x_context'].append(x_context)
            dataset_tensor['x_padding'].append(x_padding)
            dataset_tensor['freq'].append(freq)
            dataset_tensor['x_future'].append(x_future)
        elif data_type == "multivariate":
            for k in dataset_tensor:
                dataset_tensor[k].append(item[k])
        else:
            raise ValueError(f"data_type must be one of ['univariate', 'multivariate'], not {data_type}")
    print("processed dataset:", dataset_tensor.keys(), len(time_series))#, time_series[0])
        # print(x_context.shape, x_padding.shape, freq.shape, x_future.shape)
    # if data_type == "univariate":
    #     dataset_tensor['x_context'] = torch.stack(dataset_tensor['x_context'])
    #     dataset_tensor['x_padding'] = torch.stack(dataset_tensor['x_padding'])
    #     dataset_tensor['freq'] = torch.stack(dataset_tensor['freq'])
    #     dataset_tensor['x_future'] = torch.stack(dataset_tensor['x_future'])
    # elif data_type == "multivariate":
    #     for k in dataset_tmp:
    #         dataset_tmp[k] = torch.stack(dataset_tmp[k])
    #     x_context, x_padding, freq, x_future = [t for t in process_batch(tfm, dataset_tmp)]
    #     print(len(time_series), x_context.shape, x_padding.shape, freq.shape, x_future.shape)
    #     for i in range(len(time_series)):
    #         ts = torch.cat((x_context[i], x_future[i]), dim=-1).numpy()
    #         dataset_arrow.append({"start": start, "target": ts})
    #     dataset_tensor['x_context'] = x_context
    #     dataset_tensor['x_padding'] = x_padding
    #     dataset_tensor['freq'] = freq
    #     dataset_tensor['x_future'] = x_future
    # else:
    #     raise ValueError(f"data_type must be one of ['univariate', 'multivariate'], not {data_type}")

    print(f'saving dataset to {path}')
    # if not os.path.exists(path+'.arrow'):
    #     ArrowWriter(compression=compression).write_to_file(
    #         dataset_arrow,
    #         path=path+'.arrow',
    #     )
    
    if not os.path.exists(path+'.pt'):
        torch.save(dataset_tensor, path+'.pt')




