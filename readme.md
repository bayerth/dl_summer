# Machine Learning & Deep Learning Lecture Notes

This repository contains Jupyter Notebooks and Python scripts used in lectures discussing Machine Learning and Deep
Learning concepts.

## Contents:

### Jupyter Notebooks

The `notebooks/` directory contains various Jupyter notebooks organized by topic:

#### 1 Machine Learning Basics

The `notebooks/1 machin learning basics/` directory contains introductory notebooks:

- `00_python_start.ipynb` - Introduction to Python for Data Science
- `iris_exercise.ipynb` - Exercises with the Iris dataset
- `iris_tensorflow.ipynb` - Neural Networks with TensorFlow for Iris classification
- `mnist_digits_tf.ipynb` - MNIST Digits Classification using TensorFlow
- `mnist_fashion_tf.ipynb` - Fashion-MNIST Classification using TensorFlow
- `rnn_text_generation.ipynb` - Character-level RNN text generation (Shakespeare)
- `autoencoder.ipynb` - Autoencoder demonstration

> **Note:** Additional text-processing topics may be added at a later stage.

## Features:

- Hands-on demonstrations with real-world datasets
- Implementations using TensorFlow/Keras
- Instructional notebooks designed for lecture-based learning

## How to Run:

1. Clone the repository:
   ```bash
   git clone https://github.com/bayerth/ppl.git
   ```
2. Create and activate the default environment for notebooks that do not need TensorFlow:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install -U pip setuptools wheel
   python -m pip install -e .
   ```
3. Open and run the Jupyter Notebooks in the `notebooks/` directory.

### TensorFlow Environment on Apple Silicon

TensorFlow is optional and should be installed only in the dedicated `.venv312` environment. This keeps the default
`.venv` smaller and avoids mixing TensorFlow/Metal constraints with the PyTorch notebooks.

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[tf]"
```

Register both environments as Jupyter kernels if you want to switch per notebook:

```bash
source .venv/bin/activate
python -m ipykernel install --user --name dl_summer --display-name "dl_summer (.venv)"

source .venv312/bin/activate
python -m ipykernel install --user --name dl_summer_tf --display-name "dl_summer TF (.venv312)"
```

Use `dl_summer (.venv)` for PyTorch notebooks such as `torch_fashion_mnist.ipynb`. Use
`dl_summer TF (.venv312)` for TensorFlow/Keras notebooks.

## RNN Shakespeare Training

For simple RNN text-generation experiments, use the Shakespeare training runner:

```bash
scripts/run_rnn_shakespeare_train.sh --rnn-layers 64 --epochs 3 --name shakespeare_gru_test
```

The script activates `.venv` automatically. If `.venv` does not exist, it falls back to `.venv312`.

Useful experiment options:

```bash
# Small GRU baseline
scripts/run_rnn_shakespeare_train.sh \
  --rnn-layers 64 \
  --type GRU \
  --n-steps 100 \
  --epochs 5 \
  --name shakespeare_gru_64

# Two recurrent layers
scripts/run_rnn_shakespeare_train.sh \
  --rnn-layers 64,64 \
  --type GRU \
  --n-steps 100 \
  --epochs 5 \
  --name shakespeare_gru_2x64

# LSTM variant
scripts/run_rnn_shakespeare_train.sh \
  --rnn-layers 64 \
  --type LSTM \
  --n-steps 100 \
  --epochs 5 \
  --name shakespeare_lstm_64

# Embedding-based model instead of one-hot input
scripts/run_rnn_shakespeare_train.sh \
  --rnn-layers 128 \
  --embedding-dim 32 \
  --epochs 5 \
  --name shakespeare_gru_embedding
```

Models are saved to `models/` by default. Override the output directory with `--directory`.

Runner parameters:

- `--rnn-layers`: comma-separated recurrent layer sizes, for example `64` or `64,64`
- `--type`: `GRU`, `LSTM`, or `SimpleRNN`
- `--n-steps`: sequence length used for training windows
- `--epochs`: maximum number of training epochs
- `--batch-size`: training batch size
- `--embedding-dim`: optional embedding dimension; if omitted, the runner uses one-hot encoded input
- `--name`: output model name
- `--directory`: output directory for saved Keras models

### Apple Silicon Users (M1, M2, M3, ...)

Apple Silicon TensorFlow support is provided by the optional `tf` extra in `pyproject.toml`:

```bash
source .venv312/bin/activate
python -m pip install -e ".[tf]"
```

Check that TensorFlow sees the Metal device:

```bash
python - <<'PY'
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices())
print(tf.config.list_physical_devices("GPU"))
PY
```

## Shakespeare RNN Training

The notebook `notebooks/1 machin learning basics/rnn_text_generation.ipynb` demonstrates character-level text generation on the Shakespeare corpus. The same workflow is available as a standalone script for repeatable training runs.

| File | Description |
|------|-------------|
| `rnn_shakespeare_train_runner.py` | Python training script (CLI) |
| `scripts/run_rnn_shakespeare_train.sh` | Bash wrapper: activates venv, runs training (background by default) |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rnn-layers` | *(required)* | Comma-separated neuron counts per RNN layer, e.g. `128` or `64,64` |
| `--type` | `GRU` | RNN cell type: `GRU`, `LSTM`, or `SimpleRNN` |
| `--n-steps` | `100` | Sequence length (time steps) per training window |
| `--epochs` | `20` | Maximum training epochs (early stopping may stop earlier) |
| `--batch-size` | `64` | Training batch size |
| `--embedding-dim` | `None` | If set, use `Embedding` layer; otherwise one-hot encoded inputs |
| `--name` | `None` | Output model filename (without path); auto-generated if omitted |
| `--directory` | `models` | Directory for saved `.keras` models |

Auto-generated model names follow the pattern `{type}_{layers}_s{n_steps}_e{epochs}.keras`, e.g. `GRU_16_32_s100_e10.keras` for `--type GRU --rnn-layers 16,32 --n-steps 100 --epochs 10` (`s` = sequence length, `e` = max epochs).

The bash script accepts one additional flag:

| Flag | Description |
|------|-------------|
| `--foreground` | Run synchronously in the terminal (no `nohup`). Default: background via `nohup`, log under `logs/` |

The wrapper looks for a virtual environment at `.venv`, then `.venv312`.

### Example invocations

**Background training (default)** — survives terminal disconnect; output in `logs/rnn_shakespeare_train_<timestamp>.log`:

```bash
scripts/run_rnn_shakespeare_train.sh --rnn-layers 128 --type GRU --n-steps 100 --epochs 10
```

**Foreground / synchronous** — blocks until training finishes:

```bash
scripts/run_rnn_shakespeare_train.sh --foreground --rnn-layers 64,64 --type GRU --n-steps 100 --epochs 20
```

**Two-layer GRU with embedding** (saves as `models/GRU_64_64_s100_e20.keras` unless `--name` is set):

```bash
scripts/run_rnn_shakespeare_train.sh --rnn-layers 64,64 --embedding-dim 16 --batch-size 32
```

**Custom output directory and model name:**

```bash
scripts/run_rnn_shakespeare_train.sh --rnn-layers 16,32 --name my_shakespeare_model --directory trained_models
```

**Direct Python call** (with activated venv):

```bash
python rnn_shakespeare_train_runner.py --rnn-layers 128 --type GRU --n-steps 50 --epochs 10 --batch-size 64
```

**LSTM, one-hot input (default), custom save path:**

```bash
python rnn_shakespeare_train_runner.py --rnn-layers 256 --type LSTM --n-steps 100 --directory models --name shakespeare_lstm_256
```
