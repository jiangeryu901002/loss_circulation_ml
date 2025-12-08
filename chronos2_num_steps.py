import os
import torch
import numpy as np

from chronos import BaseChronosPipeline, Chronos2Pipeline
from utils import compute_metrics, load_CSM

# 只用一个 GPU（如果有的话）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def data_processing(data, target_col, past_covariate_cols, future_covariate_cols, for_training=False):
    """
    data: list of (x, y)
      - x: 过去 context_len 个时间步（DataFrame）
      - y: 未来 horizon_len 个时间步（DataFrame）
    这段与原 chronos_2cov 里的 data_processing 一致，保证输入格式相同
    """
    inputs = []
    outputs = []

    for x, y in data:
        # 目标变量：过去 + 未来
        past_target = x[target_col].to_numpy().reshape(-1)      # (context_len,)
        future_target = y[target_col].to_numpy().reshape(-1)    # (horizon_len,)

        if for_training:
            # 训练时把过去 + 未来拼在一起，交给 Chronos-2
            target_all = np.concatenate([past_target, future_target])  # (context_len + horizon_len,)
        else:
            # 评估时只给过去部分
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


def evaluation(pipeline: Chronos2Pipeline,
               val_data,
               horizon_len: int,
               save_path: str = None):
    """
    在验证集上评估，返回你关心的四个指标：MSE, MAE, SMAPE, R^2
    """
    inputs, outputs = val_data

    # Chronos-2 推理：返回 quantiles 和 mean（mean 是一个 list[Tensor]）
    quantiles, mean = pipeline.predict_quantiles(
        inputs,
        prediction_length=horizon_len,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    # 把每个 batch 的预测在 batch 维度上拼起来 => (N, horizon_len)
    preds = torch.cat(mean, dim=0).cpu().numpy()
    labels = outputs  # (N, horizon_len)

    if save_path is not None:
        np.savez(save_path, labels=labels, preds=preds)

    # 这里假设 compute_metrics 返回 [MSE, MAE, SMAPE, R2]
    # 如果你自己的 compute_metrics 顺序不一样，可以按实际情况改索引
    metrics = compute_metrics(labels, preds)
    res = {
        "MSE": float(metrics[0]),
        "MAE": float(metrics[1]),
        "SMAPE": float(metrics[2]),
        "R2": float(metrics[3]),
    }
    print("Validation metrics:", res)
    return res


def main():
    # ==== 和你原来的 chronos_2cov 设置保持一致 ====
    target_col, time_col = 'Fluid Loss', 'time_dt'
    past_covariate_cols = [
        'env', 'inclination_input', 'in_flow_rate_input', 'thruster_force_input',
        'Inclination', 'In Flow Rate', 'Thruster Force', 'Weight on Bit', 'Torque on Bit',
        'Drilling Speed', 'diff_Distance', 'diff_Depth', 'Internal Pressure', 'Annular Pressure'
    ]
    future_covariate_cols = ['inclination_input', 'in_flow_rate_input', 'thruster_force_input']

    batch_size = 128
    root_path = './data'
    file_names = ["0429_model_v5.csv", "0501_model_v5.csv", "0428_model_v5.csv"]

    # 这次只做 context_len = 128, horizon_len = 3
    context_len = 128
    horizon_len = 12

    # 你要扫描的 num_steps 列表
    num_steps_list = [1,2,3,5,10]

    # ==== 加载数据 ====
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

    data = features  # data 是一个 dict，包含 'train', 'val', 'test'
    print("Created datasets:")
    print(f"- Training samples:   {len(data['train'])}")
    print(f"- Validation samples: {len(data['val'])}")
    print(f"- Testing samples:    {len(data['test'])}")
    print(f"Using context_len={context_len}, horizon_len={horizon_len}")
    print("=" * 80)

    # 转成 Chronos-2 需要的输入格式
    train_inputs, train_outputs = data_processing(
        data['train'], target_col, past_covariate_cols, future_covariate_cols, for_training=True
    )
    val_inputs, val_outputs = data_processing(
        data['val'], target_col, past_covariate_cols, future_covariate_cols, for_training=True
    )
    # 如果以后想在 test 上评估，也可以：
    # test_inputs, test_outputs = data_processing(
    #     data['test'], target_col, past_covariate_cols, future_covariate_cols, for_training=False
    # )

    results = []

    # ==== 扫描不同的 num_steps ====
    for num_steps in num_steps_list:
        print("\n" + "#" * 80)
        print(f"Training Chronos-2 with num_steps = {num_steps}")
        print("#" * 80)

        # 每个 num_steps 都重新从预训练模型开始
        pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2",
            device_map="cuda" if device.type == "cuda" else "cpu",
        )

        # fine-tune
        finetuned_pipeline = pipeline.fit(
            inputs=train_inputs,
            validation_inputs=val_inputs,
            prediction_length=horizon_len,
            num_steps=num_steps,
            learning_rate=1e-5,
            batch_size=batch_size,
            logging_steps=10,
        )

        # 在验证集上评估
        save_name = f"./results/chronos2_c{context_len}_h{horizon_len}_steps{num_steps}.npz"
        val_metrics = evaluation(
            finetuned_pipeline,
            (val_inputs, val_outputs),
            horizon_len=horizon_len,
            save_path=save_name,
        )

        row = {"num_steps": num_steps}
        row.update(val_metrics)
        results.append(row)

    print("\n" + "=" * 80)
    print("Summary of num_steps search (on validation set):")
    for r in results:
        print(r)

    # 可选：保存成一个简单的 CSV，方便你画图 / 进论文
    # try:
    #     import pandas as pd
    #     df = pd.DataFrame(results)
    #     out_csv = f"./results/chronos2_num_steps_tuning_c{context_len}_h{horizon_len}.csv"
    #     os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    #     df.to_csv(out_csv, index=False)
    #     print(f"\nSaved summary to: {out_csv}")
    # except ImportError:
    #     print("pandas not installed, skip saving CSV.")


if __name__ == "__main__":
    main()