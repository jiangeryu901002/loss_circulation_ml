# Drilling Fluid Lost-Circulation Forecasting

This repository contains research code for forecasting drilling-fluid loss from multivariate drilling time series. It compares and adapts time-series foundation models—including Amazon Chronos/Chronos-2, Google TimesFM, and IBM Granite Tiny Time Mixers (TTM)—and includes a graph-attention extension that uses prior relationships among drilling variables.

The main prediction target is **`Fluid Loss`** (called **`Q_Loss`** in the legacy TTM pipeline). The code supports multiple context and forecast horizons, zero-shot evaluation, model fine-tuning, metric calculation, and result visualization.

> **Research-code status:** the repository currently expects private experimental data and pretrained checkpoints that are not committed to Git. Several experiment settings and paths are defined directly in the Python scripts. Read [Data preparation](#data-preparation) and [Known configuration requirements](#known-configuration-requirements) before running an experiment.

## Research workflow

The repository implements the following general workflow:

1. Load three drilling runs from CSV files.
2. Construct derived inputs and sliding windows.
3. Use two runs for shuffled training/validation samples and hold out the third run for testing.
4. Train, fine-tune, or evaluate a forecasting model.
5. Save predictions and labels as NumPy archives.
6. Calculate regression metrics and plot comparisons across models and experimental settings.

The held-out run can be changed with `fold_id`:

| `fold_id` | Training/validation runs | Test run |
|---:|---|---|
| `0` | first and second CSV | third CSV |
| `1` | second and third CSV | first CSV |
| `2` | third and first CSV | second CSV |

## Models and experiments

| Model family | Main scripts | Purpose |
|---|---|---|
| Chronos-2 | `Chronos2_cov.py`, `chronos2_num_steps.py` | Covariate-aware forecasting and fine-tuning with `amazon/chronos-2` |
| Chronos | `run_chronos.py` | Unified univariate/multivariate zero-shot and fine-tuned Chronos T5 experiments |
| TimesFM | `run_timesfm.py` | Unified univariate/multivariate TimesFM evaluation and fine-tuning |
| Granite TTM + GAT | `train_CSM_TSFM.py`, `test_CSM_TSFM.py` | TTM forecasting with a graph-attention model based on drilling-variable relationships |
| LSTM baseline | `LSTM_baseline_file_split.py` | Conventional multivariate sequence-model baseline with file-level train/test separation |
| Naive baseline | `naive baseline.py` | Persistence-style reference forecast derived from saved test labels |
| Diagnostics | `shift_test.py`, `evaluation matrix.py` | Distribution-shift checks and evaluation summaries |
| Sensitivity analysis | `run_sensitivity.py` | Single-origin, full-series, single-control, and multi-control counterfactual experiments |
| Visualization | `plot_experiments.py`, `plot_sensitivity.py` | Unified CLI for metric, forecast, and sensitivity plots |

The local `baselines/` directory contains the Chronos and TimesFM implementations used by the experiment scripts. The custom TTM/GAT architecture is defined in `modules/model_CSM_TSFM_GAT.py`.

## Repository structure

```text
.
├── baselines/                  # Local Chronos and TimesFM implementations
├── modules/
│   ├── model_CSM_TSFM_GAT.py   # Custom TTM + graph-attention model
│   ├── layers.py               # Graph/model layers
│   └── layers_ATTN.py          # Attention layers
├── sensitivity_outputs/        # Generated sensitivity CSV files and figures
├── prepare_data.py             # CSV-to-windowed-dataset preprocessing
├── datasets.py                 # PyTorch dataset and serialization utilities
├── utils.py                    # Current CSM loader, preprocessing, and metrics
├── utils_TSFM.py               # Legacy TTM data and training utilities
├── utils_causality.py          # Causality/graph utilities
├── Chronos2_cov.py             # Chronos-2 covariate experiment
├── run_chronos.py              # Unified Chronos experiment CLI
├── run_timesfm.py              # Unified TimesFM experiment CLI
├── LSTM_baseline_file_split.py # File-split multivariate LSTM baseline
├── naive baseline.py           # Persistence-style baseline evaluation
├── run_sensitivity.py          # Unified Chronos-2 sensitivity CLI
├── plot_experiments.py         # Unified metrics and forecast plotting CLI
├── plot_sensitivity.py         # Unified sensitivity plotting CLI
├── train_CSM_TSFM.py           # TTM/GAT training entry point
├── test_CSM_TSFM.py            # TTM/GAT checkpoint evaluation
└── losses.py                   # Forecasting loss functions
```

Generated data, results, and checkpoints are intentionally excluded by `.gitignore`.

## Data preparation

The current preprocessing code expects three CSV files:

```text
data/
├── 0429_model_v5.csv
├── 0501_model_v5.csv
└── 0428_model_v5.csv
```

Some TimesFM scripts instead expect the same raw files and processed tensors under `data/csm/`. Check the `root_path` variable in the selected entry point and use one location consistently.

Each input CSV must contain these columns with the exact spelling shown:

```text
Inclination
In Flow Rate
Thruster Force
Weight on Bit
Torque on Bit
Drilling Speed
diff_Distance
diff_Depth
Internal Pressure
Annular Pressure
Fluid Loss
time_dt
```

`time_dt` must be parseable by `pandas.to_datetime`. The loader additionally creates:

- `env`, a synthetic environment feature;
- rolling-mean control inputs for inclination, inflow rate, and thruster force;
- first-difference features;
- zero-padded history at the beginning of each run;
- standardized features fitted on the first two runs in the selected fold.

To generate serialized `.pt` datasets for the Chronos/TimesFM pipelines, first set `root_path`, `file_names`, context lengths, and forecast horizons in `prepare_data.py`, then run:

```bash
python prepare_data.py
```

The default settings generate univariate and multivariate train/validation/test datasets for several context/horizon combinations.

> The drilling CSV files may contain proprietary or sensitive operational data. Verify that you have permission before publishing or redistributing them.

## Environment setup

Python 3.10–3.12 is recommended. A CUDA-capable GPU is strongly recommended for foundation-model fine-tuning.

Create an isolated environment:

```bash
conda create -n lost-circulation python=3.12
conda activate lost-circulation
```

Install PyTorch using the command appropriate for your CUDA version from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). For example, the original Chronos-2 setup used CUDA 12.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Install the shared scientific Python dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib networkx tqdm huggingface-hub transformers
```

Install the packages needed by the experiment you plan to run:

```bash
# Chronos-2
pip install "chronos-forecasting>=2.0" "pandas[pyarrow]"

# IBM Granite TTM
pip install tsfm-public
```

TimesFM uses the implementation under `baselines/timesfm/`; its pretrained weights are downloaded from Hugging Face on first use. Some fine-tuning utilities also enable Weights & Biases, so install and configure `wandb` or set `use_wandb=False` in the relevant script.

## Running experiments

All commands below assume that the working directory is the repository root.

### Chronos-2 with covariates

Edit the experiment settings near the bottom of `Chronos2_cov.py`, especially `root_path`, `settings`, batch size, and fine-tuning steps, then run:

```bash
python Chronos2_cov.py
```

The script downloads `amazon/chronos-2`, fine-tunes it with past and known-future covariates, and saves label/prediction arrays under `results/`.

### Prepare datasets for Chronos and TimesFM

```bash
python prepare_data.py
```

Then select an experiment and data layout:

```bash
python run_chronos.py --data-type univariate --mode both
python run_chronos.py --data-type multivariate --mode both
python run_timesfm.py --data-type univariate --mode zeroshot
python run_timesfm.py --data-type multivariate --mode finetune
```

Use `--settings` to select context/horizon pairs, for example `--settings 32:3 32:6 32:12`. Both CLIs also expose data directories, result directories, checkpoint patterns, batch size, and model-specific options through `--help`. Their defaults reproduce the experiment selections from the former separate univariate and multivariate scripts.

### Granite TTM with graph attention

After updating the data and output paths described below, run:

```bash
python train_CSM_TSFM.py --epochs 50
python test_CSM_TSFM.py
```

The training script uses `ibm-granite/granite-timeseries-ttm-r2` and compares a standard fine-tuning route with the custom graph-attention route. The test script expects an existing checkpoint and a test CSV.

### LSTM and naive baselines

The LSTM baseline automatically finds numeric columns shared by all three CSV files. It builds windows independently within each drilling run, fits feature and target scalers on the two training runs only, and holds out the third run for testing. Review the configuration block at the top of the script, then run:

```bash
python LSTM_baseline_file_split.py
```

The default experiment uses a 128-step context, a 12-step prediction horizon, a two-layer LSTM, and early stopping. It saves train and test predictions under `results/`.

After generating Chronos-2 result archives for horizons 1, 3, 6, and 12, run the persistence-style reference baseline with:

```bash
python "naive baseline.py"
```

### Chronos-2 sensitivity analysis

The unified sensitivity CLI loads a locally fine-tuned Chronos-2 checkpoint and compares its baseline forecast with counterfactual forecasts obtained by scaling known-future control inputs. It supports inflow rate, inclination, and thruster force at one or more factors:

```bash
python run_sensitivity.py \
  --checkpoint checkpoints/chronos-2/checkpoint-final \
  --scope all \
  --targets flow inclination thruster \
  --factors 0.1 3 10
```

For a single forecast origin, use `--scope single --sample-index 500`. Other options configure the input CSV, context/horizon lengths, control-window length, batch size, output directory, device, targets, and factors. Outputs are written to `sensitivity_outputs/` by default.

These perturbation experiments measure model sensitivity; by themselves, they should not be interpreted as proof of a causal effect.

### Visualization

All experiment plots are generated through two command-line entry points. Static paper figures for prediction horizon, context length, and fine-tuning steps can be reproduced with:

```bash
python plot_experiments.py horizon
python plot_experiments.py context
python plot_experiments.py steps
```

To summarize saved `.npz` results and compare predictions across horizons:

```bash
python plot_experiments.py comparison --context 128 --horizons 1 3 6 12
python plot_experiments.py shifted --context 128 --horizons 1 3 6 12
```

Common options include `--results-dir`, `--pattern`, `--max-points`, `--output`, and `--no-show`. Run `python plot_experiments.py --help` for the complete interface.

Sensitivity plots use one implementation for all three controllable inputs:

```bash
python plot_sensitivity.py flow
python plot_sensitivity.py inclination
python plot_sensitivity.py thruster
```

Use `--factors`, `--input-dir`, `--context`, `--horizon`, `--x-min`, `--x-max`, and `--output` to reproduce customized variants. This replaces the former collection of target-specific plotting scripts without removing any plotting capability.

## Outputs and evaluation

Most forecasting scripts save compressed NumPy archives (`.npz`) containing at least:

- `labels`: ground-truth future values;
- `preds`: model forecasts;
- optional scaling statistics used to recover the original units.

The common evaluation helper reports:

- mean squared error (MSE);
- mean absolute error (MAE);
- mean absolute percentage error (MAPE);
- symmetric MAPE (sMAPE);
- root mean squared error or log-RMSE, depending on the experiment;
- coefficient of determination (R²).

`compute_metrics()` returns these values in the same order shown above. Optional 90% bootstrap confidence intervals are computed over forecast samples.

The plotting CLIs default to the repository's existing result filename conventions; use their path and pattern options when your output names differ.

## Known configuration requirements

This repository preserves experiment scripts close to their original research form. Before reproducing a run, check the following:

1. **Data are not included.** Add the three CSV files under the path expected by the selected script.
2. **Output directories may need to be created.** Create `results/`, `result/`, and `checkpoints/` as required by the entry point.
3. **Hard-coded local paths exist.** In particular, `train_CSM_TSFM.py` and `test_CSM_TSFM.py` contain a Windows temporary directory beginning with `H:/DomainAdaptation/...`; replace it with a writable local directory.
4. **Checkpoint paths are experiment-specific.** `test_CSM_TSFM.py` and the older Chronos scripts reference checkpoints from previous runs. Point them to your own checkpoints or enable the zero-shot path in the code.
5. **GPU assumptions differ.** `Chronos2_cov.py` explicitly requests CUDA. The unified legacy Chronos and sensitivity CLIs provide `--cpu`; TimesFM selects its backend from PyTorch's CUDA availability.
6. **Experiment parameters are mostly constants.** Context length, prediction horizon, fold, model variant, learning rate, and output filenames are generally edited inside each script rather than supplied as command-line arguments.
7. **Legacy naming differs.** The current CSV loader uses `Fluid Loss`, while the older TTM code uses `Q_Loss` and a different set of column names. Confirm that the selected loader matches your dataset schema.

## Reproducibility notes

The loaders set NumPy/Python seeds to 42, and the training entry point also seeds PyTorch. Exact results can still vary with GPU kernels, dependency versions, random environment features, pretrained-model revisions, and shuffled window order. For a publication-quality run, record:

- the Git commit;
- Python and package versions;
- GPU/CUDA versions;
- Hugging Face model revision;
- fold, context length, prediction horizon, and random seed;
- all local path and checkpoint changes.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Citation

If you use this code in academic work, please cite the associated paper. A BibTeX entry can be added here once the paper metadata is public:

```bibtex
@article{lost_circulation_forecasting,
  title   = {TODO: Paper title},
  author  = {TODO: Authors},
  journal = {TODO: Journal or conference},
  year    = {TODO: Year}
}
```
