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

> **Note:** Text Processing and other advanced topics will be added at a later stage.

## Features:

- Hands-on demonstrations with real-world datasets
- Implementations using TensorFlow/Keras
- Instructional notebooks designed for lecture-based learning

## How to Run:

1. Clone the repository:
   ```bash
   git clone https://github.com/bayerth/ppl.git
   ```
2. Install required Python packages using `pip`:
   ```bash
   pip install -e .
   ```
3. Open and run the Jupyter Notebooks in the `notebooks/` directory.

### Apple Silicon Users (M1, M2, M3, ...)

For Apple Silicon users, it is recommended to use:

- **Python 3.12**
- **TensorFlow 2.19**
- **TensorFlow-metal** (for GPU acceleration)

```bash
pip install tensorflow==2.19.* tensorflow-metal
```

