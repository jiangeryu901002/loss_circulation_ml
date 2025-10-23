import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))
# quit()
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness
from baselines.datasets import prepare_datasets
from momentfm.data.informer_dataset import InformerDataset
from momentfm.utils.forecasting_metrics import get_forecasting_metrics
from data_loader import load_CSM, compute_metrics, arrange_input

batch_size = 32
horizon_len, context_len = 12, 32
adj, train_loader, test_loader, data_scaler, names = load_CSM(None, normalization=None,
                                                              fold_id=2, in_len=context_len,
                                                              seq_len=horizon_len + context_len,
                                                              split_ratio=0.1, BATCH_SIZE=batch_size,
                                                              mask_head=4, cuda=torch.cuda.is_available())
# train_X, train_Y = zip(*[arrange_input(x, y, context_len, horizon_len, 1) for x, y in zip(train_loader[1], train_loader[1])])
# train_X = np.transpose(np.concatenate(train_X, axis=0), (0, 2, 1)) # sample, feature, time
# train_Y = np.transpose(np.concatenate(train_Y, axis=0), (0, 2, 1))
# train_X = torch.from_numpy(train_X).float()
# train_Y = torch.from_numpy(train_Y).float()
#
# test_X, test_Y = zip(*[arrange_input(x, y, context_len, horizon_len, 1) for x, y in zip(test_loader[1], test_loader[1])])
# test_X = np.transpose(np.concatenate(test_X, axis=0), (0, 2, 1))
# test_Y = np.transpose(np.concatenate(test_Y, axis=0), (0, 2, 1))
# test_X = torch.from_numpy(test_X).float()
# test_Y = torch.from_numpy(test_Y).float()

# print(f'train_X.shape: {train_X.shape}, train_Y.shape: {train_Y.shape}, test_X.shape: {test_X.shape}, test_Y.shape: {test_Y.shape}')

# Set random seeds for PyTorch, Numpy etc.
control_randomness(seed=13)

# Load data
train_dataset, _ = prepare_datasets(
    series=train_loader[0],
    context_length=context_len,
    horizon_length=horizon_len,
    freq_type=0,
    train_split=1,
)

test_dataset, _ = prepare_datasets(
    series=test_loader[0],
    context_length=context_len,
    horizon_length=horizon_len,
    freq_type=0,
    train_split=1,
)
# train_dataset.samples.append(test_dataset.samples[0])
# test_dataset.samples = test_dataset.samples[1:]

print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-base",
    model_kwargs={
        'task_name': 'forecasting',
        'forecast_horizon': 12,
        'head_dropout': 0.1,
        'weight_decay': 0,
        'freeze_encoder': True,  # Freeze the patch embedding layer
        'freeze_embedder': True,  # Freeze the transformer encoder
        'freeze_head': False,  # The linear forecasting head must be trained
    },
    # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
)
model.init()
print("Unfrozen parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print('    ', name)

cur_epoch = 0
max_epoch = 10
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Create a OneCycleLR scheduler
max_lr = 1e-3
total_steps = len(train_loader) * max_epoch
scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)

# Move the model to the GPU
model = model.to(device)

# Move the loss function to the GPU
criterion = criterion.to(device)

# Enable mixed precision training
scaler = torch.amp.GradScaler()

# Gradient clipping value
max_norm = 5.0

while cur_epoch < max_epoch:
    losses = []
    model.train()
    for timeseries, forecast, input_mask in tqdm(train_loader, total=len(train_loader)):
        # Move the data to the GPU
        timeseries = timeseries.float().to(device)
        input_mask = input_mask.to(device)
        forecast = forecast.float().to(device)

        with torch.amp.autocast(device_type='cuda'):
            output = model(x_enc=timeseries, input_mask=input_mask)

        loss = criterion(output.forecast, forecast)

        # Scales the loss for mixed precision training
        scaler.scale(loss).backward()

        # Clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

    losses = np.array(losses)
    average_loss = np.average(losses)
    print(f"Epoch {cur_epoch}: Train loss: {average_loss:.3f}")

    # Step the learning rate scheduler
    scheduler.step()
    cur_epoch += 1

    # Evaluate the model on the test split
    trues, preds, histories, losses = [], [], [], []
    model.eval()
    with torch.no_grad():
        for timeseries, forecast, input_mask in tqdm(test_loader, total=len(test_loader)):
            # Move the data to the GPU
            timeseries = timeseries.float().to(device)
            input_mask = input_mask.to(device)
            forecast = forecast.float().to(device)

            with torch.amp.autocast(device_type='cuda'):
                output = model(x_enc=timeseries, input_mask=input_mask)

            loss = criterion(output.forecast, forecast)
            losses.append(loss.item())

            trues.append(forecast.detach().cpu().numpy())
            preds.append(output.forecast.detach().cpu().numpy())
            histories.append(timeseries.detach().cpu().numpy())

    losses = np.array(losses)
    average_loss = np.average(losses)
    trues = np.concatenate(trues, axis=0)
    preds = np.concatenate(preds, axis=0)
    histories = np.concatenate(histories, axis=0)

    metrics = compute_metrics(ys=trues, preds=preds, scaler=data_scaler[0].scale_)#, cum=True)
    print(f"Epoch {cur_epoch}:")
    # print(metrics[0].shape, metrics)
    for i, n in enumerate(names[3:-3]):
        print(
            f"{n}: MSE: {metrics[0][:, i + 3]} | MAE: {metrics[1][:, i + 3]} "
            f"| MAPE: {metrics[2][:, i + 3]} | R^2: {metrics[3][:, i + 3]}")
