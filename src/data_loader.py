import os
import urllib.request

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MNIST_PATH = os.path.join(DATA_DIR, "mnist.npz")


def _download_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(MNIST_PATH):
        print(f"Downloading MNIST to {MNIST_PATH} ...")
        urllib.request.urlretrieve(MNIST_URL, MNIST_PATH)
        print("Download complete.")


def _one_hot(labels, num_classes=10):
    # labels: (N,) int -> one_hot: (num_classes, N), matching mlp_network's column-vector targets
    one_hot = np.zeros((num_classes, labels.shape[0]), dtype=np.float32)
    one_hot[labels, np.arange(labels.shape[0])] = 1.0
    return one_hot


def load_mnist():
    _download_mnist()

    with np.load(MNIST_PATH) as data:
        x_train_raw = data["x_train"]
        y_train_raw = data["y_train"]
        x_test_raw = data["x_test"]
        y_test_raw = data["y_test"]

    # (N, 28, 28) uint8 -> (N, 1, 28, 28) float32 normalized to [0, 1]
    x_train = (x_train_raw[:, np.newaxis, :, :] / 255.0).astype(np.float32)
    x_test = (x_test_raw[:, np.newaxis, :, :] / 255.0).astype(np.float32)

    # (N,) int labels -> (10, N) one-hot columns
    y_train = _one_hot(y_train_raw)
    y_test = _one_hot(y_test_raw)

    return x_train, y_train, x_test, y_test
