import os.path
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
try:
    from .dataset.arrow import ArrowWriter
except:
    pass
from utils import process_batch
from baselines.timesfm import TimesFm

class TimeSeriesDataset(Dataset):
    """Dataset for time series data compatible with TimesFM."""

    def __init__(self,
                scaler,
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
        self.x_context = series['x_context']
        self.x_future = series['x_future']
        self.x_padding = series['x_padding']
        self.freq = series['freq']
        self.scaler_target = scaler

    def __len__(self) -> int:
        return self.x_context.shape[0]

    def __getitem__(self, index: int):
        x_context, input_padding, freq, x_future = self.x_context[index], self.x_padding[index], self.freq[index], self.x_future[index]
        return x_context, input_padding, freq, x_future  # x_context, x_future, input_mask #



class Dataset_Custom(Dataset):

    def __init__(
        self,
        root_path,
        data_path,
        flag='train',         # 'train', 'val', or 'test'
        size=None,            # [seq_len, pred_len]
        features='MS',        # 多变量 -> 单输出
        target='price',       # 目标列
        freq=0,             # 月度(这里不实际用, 但可用于 embed.py)
        train_ratio=0.7,      # 每个zipcode训练集比例
        val_ratio=0.33,       # 剩余中的验证集比例
        scale=True,
        data_type='univariate', # or multivariate
    ):
        super().__init__()

        # 1) 滑窗相关
        if size is None:
            self.context_len = 6
            self.horizon_len = 6
        else:
            self.context_len, self.horizon_len = size

        self.flag = flag  # 'train','val','test'
        self.features = features
        self.target = target
        self.freq = freq
        self.scale = scale
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_type

        # 存放每个zipcode的数据块
        self.series_list = []   # 普通特征 X
        self.out_list = []      # 目标列  Y
        self.xmark_list = []    # 存放 year, month 等时间戳特征

        # index_map 记录 (series_idx, start_pos)
        self.index_map = []
        self.scaler_cov, self.scaler_target = None, None

        self.__read_data__()

    def __read_data__(self):
        fpath = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(fpath, parse_dates=['date'])
        # 按 (zipcode, date) 排序
        df_raw = df_raw.sort_values(by=['zipcode','date']).reset_index(drop=True)

        # =============== 新增：在df里加 year, month 列 ===============
        df_raw['year'] = df_raw['date'].dt.year
        df_raw['month'] = df_raw['date'].dt.month

        # 你也可以类似 df_raw['day'] = df_raw['date'].dt.day etc.
        # 这里只是示例演示 year/month 两列

        # 去掉不需要列
        drop_cols = ['city','city_full']
        df_raw.drop(columns=drop_cols, inplace=True, errors='ignore')

        # =============== 列分配 ===============
        # 我们把 year/month 视为时间戳特征
        time_mark_cols = ['year','month']

        # 把 date, zipcode, target, year, month 全剔除后剩余的就是普通数值特征
        remove_list = ['date','zipcode', self.target] + time_mark_cols
        all_cols = list(df_raw.columns)
        numeric_cols = [c for c in all_cols if c not in remove_list]

        # 全局 scaler_cov
        scale_data = []

        grouped = df_raw.groupby("zipcode")
        for zipcode, subdf in grouped:
            subdf = subdf.sort_values("date")
            N = len(subdf)
            # 如果长度小于 seq_len+pred_len，跳过
            if N < (self.context_len + self.horizon_len):
                continue

            # 按比例切分
            train_count = int(N * self.train_ratio)
            remain = N - train_count
            val_count = int(N * self.val_ratio)
            test_count = remain - val_count

            # split by time for each county
            if self.flag == 'train':
                sub_data = subdf.iloc[:train_count]
            elif self.flag == 'val':
                sub_data = subdf.iloc[train_count : train_count + val_count]
            elif self.flag == 'test':
                sub_data = subdf.iloc[train_count + val_count : train_count + val_count + test_count]
            else:
                raise ValueError("flag 必须是 'train'/'val'/'test'.")

            n_sub = len(sub_data)
            if n_sub < (self.context_len + self.horizon_len):
                continue

            scale_data.append(subdf.iloc[:train_count])

            # X / Y / X_mark
            X = sub_data[numeric_cols]  # shape [n_sub, d_in], d_in=34
            Y = np.log1p(sub_data[self.target].values.reshape(-1, 1))  # shape [n_sub, 1]
            X_mark = sub_data[time_mark_cols].values      # shape [n_sub, 2]

            # 记录
            series_idx = len(self.series_list)
            self.series_list.append(X)
            self.out_list.append(Y)
            self.xmark_list.append(X_mark)

            # 构建滑窗 index
            length = X.shape[0]
            max_start = length - (self.context_len + self.horizon_len)
            for stpos in range(max_start + 1):
                self.index_map.append((series_idx, stpos))

        scale_data = pd.concat(scale_data)
        prices = np.log1p(scale_data[[self.target]].values)
        if self.scale:
            self.scaler_cov = StandardScaler()
            self.scaler_cov.fit(scale_data[numeric_cols].values)
            self.scaler_target = StandardScaler()
            self.scaler_target.fit(prices)

        for i in range(len(self.series_list)):
            X, Y = self.series_list[i], self.out_list[i]
            if self.scaler_cov:
                X = pd.DataFrame(self.scaler_cov.transform(X.values), columns=numeric_cols)
            if self.scaler_target:
                Y = self.scaler_target.transform(Y).flatten()
            self.series_list[i], self.out_list[i] = X, Y



    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        series_idx, start_pos = self.index_map[idx]
        X = self.series_list[series_idx]      # shape [N, 34]
        Y = self.out_list[series_idx]         # shape [N, 1]
        # X_mark = self.xmark_list[series_idx]  # shape [N, 2]
        # encoder input
        seq_x = torch.from_numpy(Y[start_pos : start_pos + self.context_len]).float()           # [seq_len, 34]

        # decoder label
        r_begin = start_pos + self.context_len
        r_end   = r_begin + self.horizon_len
        seq_y   = Y[r_begin : r_end]           # [pred_len, 1]
        seq_y      = torch.from_numpy(seq_y).float()

        input_padding = torch.zeros_like(seq_x)
        freq = torch.tensor([self.freq], dtype=torch.long)

        # seq_x_mark = X_mark[start_pos : start_pos + self.context_len] # [seq_len, 2]
        # seq_y_mark = X_mark[r_begin : r_end]   # [pred_len, 2]
        # seq_x_mark = torch.from_numpy(seq_x_mark).float()
        # seq_y_mark = torch.from_numpy(seq_y_mark).float()

        if self.data_type == 'multivariate':
            covs = X.iloc[start_pos:r_end]
            covs = {c:torch.from_numpy(covs[c].values).float() for c in X.columns}
            covs['inputs'] = seq_x
            covs['outputs'] = seq_y
            covs['input_padding'] = input_padding
            covs['freq'] = freq
            return covs #seq_x, seq_y, seq_x_mark, seq_y_mark, covs
        else:
            if seq_x.shape[0] < 32:
                padding = torch.zeros((32 - seq_x.shape[0], *seq_x.shape[1:]))
                seq_x = torch.concat([padding, seq_x])
                input_padding = torch.concat([torch.ones_like(padding), input_padding])
            return seq_x, input_padding, freq, seq_y

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
    tfm: TimesFm,
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
    dataset_arrow, dataset_tensor = [], {'x_context':[], 'x_padding':[], 'freq':[], 'x_future':[]}
    dataset_tmp = {k:[] for k in time_series[0]}
    for item in time_series:
        if data_type == "univariate":
            x_context, x_padding, freq, x_future = item
            ts = torch.cat((x_context, x_future), dim=-1).numpy()
            dataset_arrow.append({"start": start, "target": ts})
            dataset_tensor['x_context'].append(x_context)
            dataset_tensor['x_padding'].append(x_padding)
            dataset_tensor['freq'].append(freq)
            dataset_tensor['x_future'].append(x_future)
        elif data_type == "multivariate":
            for k in dataset_tmp:
                dataset_tmp[k].append(item[k])
        else:
            raise ValueError(f"data_type must be one of ['univariate', 'multivariate'], not {data_type}")

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
    print(time_series.scaler_target.mean_, time_series.scaler_target.scale_)
    # if not os.path.exists(path+'.arrow'):
    #     ArrowWriter(compression=compression).write_to_file(
    #         dataset_arrow,
    #         path=path+'.arrow',
    #     )
    #
    # if not os.path.exists(path+'.pt'):
    #     torch.save(dataset_tensor, path+'.pt')




