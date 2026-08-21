# Drilling Fluid Lost-Circulation Forecasting

This repository contains research code for predicting drilling-fluid lost circulation from multivariate drilling time series.

## Recommended version

> **For the cleaned, documented, and consolidated version of this project, use the [`loss_circulation_final` branch](https://github.com/jiangeryu901002/loss_circulation_ml/tree/loss_circulation_final).**

The `main` branch is retained as the original research-code snapshot so that earlier experiments and file references remain reproducible. New users should start from `loss_circulation_final`.

## Branch differences

| Area | `main` | `loss_circulation_final` |
|---|---|---|
| Purpose | Original experimental snapshot | Recommended maintained version |
| Documentation | Minimal setup notes | Complete project, data, environment, execution, and reproducibility guide |
| Chronos experiments | Separate univariate and multivariate scripts | Unified `run_chronos.py` CLI |
| TimesFM experiments | Separate univariate and multivariate scripts | Unified `run_timesfm.py` CLI |
| Chronos-2 sensitivity analysis | Experiment-specific scripts | Unified `run_sensitivity.py` CLI |
| Plotting | Multiple duplicated plotting scripts | Unified `plot_experiments.py` and `plot_sensitivity.py` CLIs |
| Baselines | Foundation-model experiments | Adds LSTM and naive persistence baselines |
| Evaluation | Original metric implementation | Consistent metric ordering and optional bootstrap confidence intervals |
| Repository hygiene | Research-oriented filenames and layout | Reduced duplication, standardized entry points, and stronger ignore rules |

The final branch preserves the original forecasting capabilities while exposing experiment choices through command-line arguments instead of requiring multiple near-duplicate scripts.

## Open the final branch

- [Browse the recommended branch](https://github.com/jiangeryu901002/loss_circulation_ml/tree/loss_circulation_final)
- [Read its complete README](https://github.com/jiangeryu901002/loss_circulation_ml/blob/loss_circulation_final/README.md)
- Clone that branch directly:

```bash
git clone --branch loss_circulation_final --single-branch \
  https://github.com/jiangeryu901002/loss_circulation_ml.git
```

If the repository is already cloned:

```bash
git fetch origin
git switch loss_circulation_final
```

## Legacy Chronos-2 quick start (`main` only)

The original minimal setup instructions are retained below for users reproducing the historical `main` branch.

```bash
conda create -n chronos python=3.12
conda activate chronos
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "chronos-forecasting>=2.0" "pandas[pyarrow]" matplotlib
python Chronos2_cov.py
```
