import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))
import torch
from baselines.timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from datasets import Dataset_Custom, convert_to_processed_dataset

def get_model(context_len, horizon_len):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = "google/timesfm-2.0-500m-pytorch"
    hparams = TimesFmHparams(
        backend=device,
        per_core_batch_size=128,
        horizon_len=horizon_len,
        num_layers=50,
        use_positional_embedding=True,
        context_len=context_len,  # Context length can be anything up to 2048 in multiples of 32
    )
    tfm = TimesFm(hparams=hparams,
                  checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id))

    return tfm


batch_size = 128
root_path = './data'

settings = [(6, 3), (6, 6), (6, 12), (12, 3), (12, 6), (12, 12)]
for context_len, horizon_len in settings:
    train_dataset = Dataset_Custom(
        root_path=root_path,
        data_path="HouseTS_log_features.csv",
        flag='train',
        size=[context_len, horizon_len],
        train_ratio=0.5,
        val_ratio=0.2,
        )

    val_dataset = Dataset_Custom(
        root_path=root_path,
        data_path="HouseTS_log_features.csv",
        flag='val',
        size=[context_len, horizon_len],
        train_ratio=0.5,
        val_ratio=0.2,
        )

    test_dataset = Dataset_Custom(
        root_path=root_path,
        data_path="HouseTS_log_features.csv",
        flag='test',
        size=[context_len, horizon_len],
        train_ratio=0.5,
        val_ratio=0.2,
        )

    print(f"Created datasets:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    print(f"- Testing samples: {len(test_dataset)}")

    tfm = get_model(context_len, horizon_len)

    data_path = f"./data/univariate_c{context_len}h{horizon_len}"
    convert_to_processed_dataset(data_path + "_train", time_series=train_dataset, tfm=tfm, data_type='univariate')
    convert_to_processed_dataset(data_path + "_val", time_series=val_dataset, tfm=tfm, data_type='univariate')
    convert_to_processed_dataset(data_path + "_test", time_series=test_dataset, tfm=tfm, data_type='univariate')

for context_len, horizon_len in settings:
    train_dataset = Dataset_Custom(
        root_path=root_path,
        data_path="HouseTS_log_features.csv",
        flag='train',
        size=[context_len, horizon_len],
        train_ratio=0.5,
        val_ratio=0.2,
        data_type='multivariate'
    )

    val_dataset = Dataset_Custom(
        root_path=root_path,
        data_path="HouseTS_log_features.csv",
        flag='val',
        size=[context_len, horizon_len],
        train_ratio=0.5,
        val_ratio=0.2,
        data_type='multivariate'
    )

    test_dataset = Dataset_Custom(
        root_path=root_path,
        data_path="HouseTS_log_features.csv",
        flag='test',
        size=[context_len, horizon_len],
        train_ratio=0.5,
        val_ratio=0.2,
        data_type='multivariate'
    )

    print(f"Created datasets:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    print(f"- Testing samples: {len(test_dataset)}")

    tfm = get_model(context_len, horizon_len)

    data_path = f"./data/multivariate_c{context_len}h{horizon_len}"
    convert_to_processed_dataset(data_path + "_train", time_series=train_dataset, tfm=tfm, data_type='multivariate')
    convert_to_processed_dataset(data_path + "_val", time_series=val_dataset, tfm=tfm, data_type='multivariate')
    convert_to_processed_dataset(data_path + "_test", time_series=test_dataset, tfm=tfm, data_type='multivariate')


