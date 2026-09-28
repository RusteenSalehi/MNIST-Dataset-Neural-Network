# MNIST From Scratch

An MNIST digit classifier implemented entirely in NumPy, with no deep
learning framework: an MLP and a CNN, each with forward pass, backpropagation,
and mini-batch gradient descent written by hand.

## Overview

This project trains and evaluates two neural networks on the MNIST
handwritten digit dataset:

- A multi-layer perceptron (`src/mlp_network.py`)
- A convolutional neural network (`src/cnn_network.py`)

Neither uses PyTorch, TensorFlow/Keras, or any other autodiff framework for
the network code. The goal is to implement the forward pass, backpropagation,
and gradient descent update rules directly, to understand the mechanics that
frameworks normally hide.

## Results

| Model | Test accuracy | Parameters | Time / epoch | Dataset |
|---|---|---|---|---|
| CNN | 97.15% (epoch 3, 2,000-sample test slice) | 27,562 | ~145-151s | Full MNIST (60,000 train / 2,000 test slice) |
| MLP | Not benchmarked in this environment | 108,830 | N/A | N/A |

CNN numbers are from a real run of `src/cnn_test.py` in this environment
(3 epochs, full 60,000-image training set, learning rate 0.05, batch size 32,
CPU only):

| Epoch | Avg. loss | Train accuracy | Test accuracy (2,000 samples) | Time |
|---|---|---|---|---|
| 1 | 0.2343 | 92.84% | 94.90% | 150.8s |
| 2 | 0.0826 | 97.37% | 97.25% | 145.1s |
| 3 | 0.0607 | 98.16% | 97.15% | 144.9s |

Test accuracy is measured on a 2,000-image slice of the test set, not the
full 10,000, because `cnn_test.py` only evaluates a slice each epoch for
speed (see `src/cnn_test.py`).

The MLP (`src/mlp_network.py`) is implemented and passes no automated check
of its own, but its training script (`src/mlp_test.py`) depends on
`keras.datasets.mnist` for data loading, which is not installed in this
project's environment (see [Project structure](#project-structure)). It has
not been run or benchmarked here.

## Architecture

### MLP (`src/mlp_network.py`)

Fully connected, three layers, ReLU activations, softmax output. Weights use
He initialization.

| Layer | Shape | Parameters |
|---|---|---|
| Input | (784, batch) | - |
| Dense 1 + ReLU | 784 -> 128 | 100,480 |
| Dense 2 + ReLU | 128 -> 60 | 7,740 |
| Dense 3 + Softmax | 60 -> 10 | 610 |
| **Total** | | **108,830** |

### CNN (`src/cnn_network.py`)

| Layer | Output shape | Parameters |
|---|---|---|
| Input | (batch, 1, 28, 28) | - |
| Conv2D(1 -> 8, 3x3) + ReLU | (batch, 8, 26, 26) | 80 |
| MaxPool2D(2, 2) | (batch, 8, 13, 13) | 0 |
| Conv2D(8 -> 16, 3x3) + ReLU | (batch, 16, 11, 11) | 1,168 |
| MaxPool2D(2, 2) | (batch, 16, 5, 5) | 0 |
| Flatten | (400, batch) | 0 |
| Dense 1 + ReLU | 400 -> 64 | 25,664 |
| Dense 2 + Softmax | 64 -> 10 | 650 |
| **Total** | | **27,562** |

Both convolutions use valid (no padding) convolution with stride 1.

## What's implemented from scratch

- 2D convolution via im2col / col2im (`src/cnn_network.py`)
- Max pooling with cached argmax indices for backward (`src/cnn_network.py`)
- Backpropagation for every layer (Dense, Conv2D, ReLU, MaxPool2D, Flatten,
  softmax + cross-entropy)
- He weight initialization
- Softmax activation with cross-entropy loss
- Mini-batch stochastic gradient descent
- Numerical gradient checking, used to validate the analytic backward passes
  (`src/gradient_check.py`)

## Project structure

```
MNIST-Dataset-Nueral-Network/
├── data/                  MNIST .npz cache (downloaded automatically, gitignored)
├── src/
│   ├── cnn_network.py     Conv2D/ReLU/MaxPool2D/Dense/SoftmaxCrossEntropy layers and the CNN
│   ├── cnn_test.py        Trains and evaluates the CNN on MNIST
│   ├── data_loader.py     Downloads and loads MNIST as normalized, one-hot arrays
│   ├── gradient_check.py  Numerical gradient checks for Conv2D and Dense
│   ├── main.py            Legacy, fully commented-out early MLP training loop (kept for history, not runnable)
│   ├── mlp_network.py     Dense-layer MLP: forward pass, backprop, softmax/cross-entropy
│   ├── mlp_test.py        Trains the MLP; requires keras for data loading (not in requirements.txt), not verified in this environment
│   └── smallExample.py    Standalone scratch script: gradient descent on a single linear neuron, unrelated to MNIST
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

- Python 3.11 or 3.12. NumPy failed to install/import on Python 3.14 during
  testing for this project; 3.11 or 3.12 is recommended.
- See `requirements.txt` for pinned package versions (NumPy only, for the
  CNN/MLP network code, data loading, and gradient checking).
- `src/mlp_test.py` additionally requires `keras` (and a backend such as
  TensorFlow) for `keras.datasets.mnist`. This is not listed in
  `requirements.txt` and was not installed or tested in this project's
  environment.

## Setup

Clone the repository and create a virtual environment:

```
git clone https://github.com/RusteenSalehi/MNIST-Dataset-Neural-Network.git
cd MNIST-Dataset-Neural-Network
python -m venv .venv
```

Activate the virtual environment.

Windows (PowerShell):

```
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```
source .venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

## Usage

All commands below are run from the `src/` directory.

### Train and evaluate the CNN

```
python cnn_test.py
```

By default this trains on the full 60,000-image MNIST training set for 3
epochs (`epochs = 3`, `batch_size = 32`, `learning_rate = 0.05`), evaluating
on a 2,000-image slice of the test set each epoch. Expect roughly 2.5 minutes
per epoch on CPU. Example output:

```
Epoch 1/3
  Average loss: 0.2343
  Train accuracy: 92.84%
  Test accuracy (2000 samples): 94.90%
  Time: 150.8s
==============================
```

To iterate quickly instead, open `cnn_test.py` and set `USE_SUBSET = True`
near the top of the file (this trains on `SUBSET_SIZE` images, 5,000 by
default, and evaluates on a 1,000-image test slice). This is a constant in
the script, not a command-line flag.

MNIST is downloaded automatically on first run (from a Google-hosted mirror
of the dataset) and cached at `data/mnist.npz` relative to the repository
root. Subsequent runs reuse the cached file.

### Run the gradient check

```
python gradient_check.py
```

This runs a `Conv2D` layer and a `Dense` layer forward and backward on small
random inputs, compares the analytic gradients from `backward()` against
numerical gradients computed by central differences, and prints the relative
error for each parameter (`d_W`, `d_b`) and for the input gradient (`d_x`).
Expected output is six relative error values, each on the order of `1e-10` or
smaller:

```
Conv2D d_W relative error: 2.6001540081358194e-11
Conv2D d_b relative error: 2.7774725039381208e-11
Conv2D d_x relative error: 6.890172270247205e-11
Dense d_W relative error: 2.0612016250650684e-11
Dense d_b relative error: 1.6496110944476608e-11
Dense d_x relative error: 2.489781002842208e-11
```

### MLP

`mlp_test.py` requires `keras` for data loading, which is not part of this
project's dependencies (see [Requirements](#requirements)). It was not run
or verified as part of this README.

## Implementation notes

**im2col/col2im.** `cnn_network.py` includes `naive_convolve2d`, a direct
nested-loop convolution, for reference only; it is not used by the `Conv2D`
layer. `Conv2D` instead uses `im2col` to unfold every convolution window into
a column of a single matrix, so that the convolution becomes one matrix
multiplication (`W_flat @ col`) instead of many small dot products in Python.
`col2im` reverses this for the backward pass, scattering column gradients
back into the overlapping input windows. This trades some memory (the
unfolded columns) for far fewer Python-level operations, since NumPy's matrix
multiplication runs in optimized, vectorized code rather than interpreted
Python loops.

**Gradient checking.** `gradient_check.py` perturbs each parameter by a small
epsilon and computes the resulting change in loss via central differences,
which approximates the true gradient independent of the hand-derived
backward pass. Comparing this numerical gradient to the analytic gradient
from `backward()` and finding a relative error near machine precision (as
above) is strong evidence that the backward pass is implemented correctly.

## Limitations and possible extensions

- CPU only; no GPU support.
- Plain mini-batch SGD with a fixed learning rate; no momentum, Adam, weight
  decay, or learning rate scheduling.
- `im2col`/`col2im` and `MaxPool2D` use Python-level loops over batch and
  output positions rather than fully vectorized NumPy operations, which is
  why a full CNN epoch takes minutes rather than seconds.
- No convolution padding, dilation, or stride other than 1 for `Conv2D`.
- No data augmentation, dropout, batch normalization, or other regularization.
- The MLP training path (`mlp_test.py`) currently depends on `keras` for data
  loading and has not been verified against this repository's own
  `data_loader.py`.

## Acknowledgments

MNIST dataset: Y. LeCun, C. Cortes, and C. J. C. Burges,
"The MNIST Database of Handwritten Digits."

## License

MIT. See [LICENSE](LICENSE).
