# Heart Disease Classification with Optuna Hyperparameter Search

A concise pipeline for training and tuning various classifiers on a heart‐disease dataset using Optuna. It handles missing values, encoding, scaling, feature selection, sampling and GPU acceleration (for supported models).

## Requirements

- Python 3.11+
- CUDA‐enabled GPU (for XGBoost GPU support) or CPU only
- uv (https://docs.astral.sh/uv/)

```bash
uv sync
```

## Train Models

```bash
python train.py --model <model_name> [--trials N] [--n-jobs N] [--data-leak <bool>]
```

## Tuning minimise penalty to recreate paper metrics

```bash
python tmp.py
```
