import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils import compute_metrics  # 复用你现有指标函数

# -----------------------------
# Config
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ROOT_PATH = "./data"
TRAIN_FILES = ["0429_model_v5.csv", "0501_model_v5.csv"]
TEST_FILES  = ["0428_model_v5.csv"]

TIME_COL   = "time_dt"
TARGET_COL = "Fluid Loss"

CONTEXT_LEN = 128
HORIZON_LEN = 12
STRIDE      = 1

BATCH_SIZE  = 128
TRAIN_RATIO = 0.8

# LSTM baseline hyperparams
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.1
EPOCHS      = 200
LR          = 1e-3
PATIENCE    = 8

RESULT_DIR  = "./results"


# -----------------------------
GROUNDWATER_ALIASES = [
    "below ground water",
    "ground water level",
    "ground water lever",
]

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in GROUNDWATER_ALIASES:
        if c in df.columns:
            df = df.rename(columns={c: "groundwater"})
            break
    return df

def sort_by_time_if_possible(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col is None or time_col not in df.columns:
        return df
    # time_dt 是 object；尝试 parse 并排序，失败则保持原顺序
    t = pd.to_datetime(df[time_col], errors="coerce")
    if t.notna().sum() > 0:
        df = df.assign(_t=t).sort_values("_t").drop(columns=["_t"]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df

# -----------------------------
# Sliding windows per file (no cross-file windows)
# -----------------------------
def build_windows_from_df(df: pd.DataFrame,
                          context_len: int,
                          horizon_len: int,
                          target_col: str,
                          feature_cols: list,
                          stride: int = 1):
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df[target_col].to_numpy(dtype=np.float32)

    T = len(df)
    max_start = T - context_len - horizon_len + 1
    if max_start <= 0:
        return None, None

    X_list, y_list = [], []
    for s in range(0, max_start, stride):
        e = s + context_len
        h = e + horizon_len
        X_list.append(X_all[s:e, :])   # [context, F]
        y_list.append(y_all[e:h])      # [horizon]
    return np.stack(X_list, axis=0), np.stack(y_list, axis=0)

def read_and_prepare(fp: str, time_col: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df = harmonize_columns(df)
    df = sort_by_time_if_possible(df, time_col)
    return df

def infer_common_feature_cols(dfs: list, target_col: str, time_col: str):
    """
    取所有文件的“数值列”交集，排除 target/time
    """
    numeric_sets = []
    for df in dfs:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        numeric_sets.append(set(numeric_cols))
    common = set.intersection(*numeric_sets)
    common.discard(target_col)
    if time_col in common:
        common.discard(time_col)
    # 排序只是为了稳定输出
    return sorted(list(common))

def load_windows_from_files(file_paths,
                            context_len,
                            horizon_len,
                            target_col,
                            feature_cols,
                            time_col=None,
                            stride=1):
    Xs, ys = [], []
    for fp in file_paths:
        df = read_and_prepare(fp, time_col)

        # 再确认列都在（防止某文件缺列）
        missing = [c for c in feature_cols + [target_col] if c not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(fp)} missing columns: {missing}")

        X, y = build_windows_from_df(df, context_len, horizon_len, target_col, feature_cols, stride=stride)
        if X is None:
            raise ValueError(f"File too short for windows: {fp}")
        Xs.append(X)
        ys.append(y)

    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

# -----------------------------
# Dataset / Model
# -----------------------------
class NumpyWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, horizon_len):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon_len)
        )
    def forward(self, x):
        out, _ = self.lstm(x)      # [B, T, H]
        last = out[:, -1, :]       # [B, H]
        return self.head(last)     # [B, horizon]

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / max(n, 1)

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn):
    model.eval()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / max(n, 1)

@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    preds, labels = [], []
    for X, y in loader:
        X = X.to(device)
        pred = model(X).cpu().numpy()
        preds.append(pred)
        labels.append(y.numpy())
    return np.concatenate(labels, axis=0), np.concatenate(preds, axis=0)

def evaluation(model, loader, save_path):
    labels, preds = predict_all(model, loader)
    np.savez(save_path, labels=labels, preds=preds)

    metrics = compute_metrics(labels, preds)
    res = {
        'MSE': metrics[0],
        'MAE': metrics[1],
        'MAPE': metrics[2],
        'SMAPE': metrics[3],
        'RMSE': metrics[4],
        'r2': metrics[5]
    }
    print(res)
    return res

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    train_paths = [os.path.join(ROOT_PATH, f) for f in TRAIN_FILES]
    test_paths  = [os.path.join(ROOT_PATH, f) for f in TEST_FILES]

    # 1) 读入文件（先做列名统一），推断 train/test 通用特征列（数值列交集）
    dfs_for_schema = []
    for fp in train_paths + test_paths:
        df = read_and_prepare(fp, TIME_COL)
        dfs_for_schema.append(df)

    feature_cols = infer_common_feature_cols(dfs_for_schema, TARGET_COL, TIME_COL)
    if len(feature_cols) == 0:
        raise ValueError("No common numeric feature columns found after harmonization.")
    print("Using feature columns (common numeric intersection):")
    print(feature_cols)

    # 2) 文件级构造 windows（不跨文件）
    X_train_all, y_train_all = load_windows_from_files(
        train_paths, CONTEXT_LEN, HORIZON_LEN, TARGET_COL, feature_cols, time_col=TIME_COL, stride=STRIDE
    )
    X_test, y_test = load_windows_from_files(
        test_paths,  CONTEXT_LEN, HORIZON_LEN, TARGET_COL, feature_cols, time_col=TIME_COL, stride=STRIDE
    )

    # 3) train windows 内部切 train/val（时间顺序）
    N = X_train_all.shape[0]
    n_tr = int(N * TRAIN_RATIO)

    X_train, y_train = X_train_all[:n_tr], y_train_all[:n_tr]
    X_val,   y_val   = X_train_all[n_tr:], y_train_all[n_tr:]

    print("Windows created:")
    print(f"- Train windows: {len(X_train)}")
    print(f"- Val windows  : {len(X_val)}")
    print(f"- Test windows : {len(X_test)}")

    # 4) 标准化（只用 train 拟合，避免泄漏）
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    y_scaler.fit(y_train.reshape(-1, 1))

    def scale_X(X):
        return x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    def scale_y(y):
        return y_scaler.transform(y.reshape(-1, 1)).reshape(y.shape)

    X_train_s = scale_X(X_train)
    X_val_s   = scale_X(X_val)
    X_test_s  = scale_X(X_test)

    y_train_s = scale_y(y_train)
    y_val_s   = scale_y(y_val)
    y_test_s  = scale_y(y_test)

    # 5) DataLoaders
    dl_train = DataLoader(NumpyWindowDataset(X_train_s, y_train_s), batch_size=BATCH_SIZE, shuffle=True)
    dl_val   = DataLoader(NumpyWindowDataset(X_val_s,   y_val_s),   batch_size=BATCH_SIZE, shuffle=False)
    dl_test  = DataLoader(NumpyWindowDataset(X_test_s,  y_test_s),  batch_size=BATCH_SIZE, shuffle=False)

    # 6) Train + early stopping
    model = LSTMForecaster(
        input_size=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon_len=HORIZON_LEN
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, dl_train, optimizer, loss_fn)
        va_loss = eval_one_epoch(model, dl_val, loss_fn)

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        print(f"Epoch {ep:02d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} | best_val={best_val:.6f}")

        if bad >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 7) Save results in Chronos-like format
    os.makedirs(RESULT_DIR, exist_ok=True)
    evaluation(model, dl_train, os.path.join(RESULT_DIR, f"train_LSTM_baseline_c{CONTEXT_LEN}h{HORIZON_LEN}.npz"))
    evaluation(model, dl_test,  os.path.join(RESULT_DIR, f"LSTM_baseline_c{CONTEXT_LEN}h{HORIZON_LEN}.npz"))

