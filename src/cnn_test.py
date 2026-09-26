import time

import numpy as np

from cnn_network import Network
from data_loader import load_mnist

# Set True to train on a small subset for fast iteration; False for a full run.
USE_SUBSET = False
SUBSET_SIZE = 5000

learning_rate = 0.05
epochs = 3
batch_size = 32

x_train, y_train, x_test, y_test = load_mnist()

if USE_SUBSET:
    x_train = x_train[:SUBSET_SIZE]
    y_train = y_train[:, :SUBSET_SIZE]

num_train = x_train.shape[0]

# Evaluating the full test set every epoch is slow with the naive im2col
# loops, so only evaluate on a slice; raise this for a final, careful number.
test_eval_size = 1000 if USE_SUBSET else 2000
x_test_eval = x_test[:test_eval_size]
y_test_eval = y_test[:, :test_eval_size]

net = Network()

for epoch in range(epochs):
    start_time = time.time()

    indices = np.random.permutation(num_train)
    total_loss = 0.0
    correct = 0

    for start in range(0, num_train, batch_size):
        batch_indices = indices[start:start + batch_size]
        x_batch = x_train[batch_indices]
        y_batch = y_train[:, batch_indices]
        actual_batch_size = x_batch.shape[0]

        probs = net.forward(x_batch)
        loss = net.loss(probs, y_batch)
        total_loss += loss * actual_batch_size

        predictions = np.argmax(probs, axis=0)
        targets = np.argmax(y_batch, axis=0)
        correct += np.sum(predictions == targets)

        net.backward(y_batch, learning_rate)

    average_loss = total_loss / num_train
    train_accuracy = correct / num_train * 100

    test_probs = net.forward(x_test_eval)
    test_predictions = np.argmax(test_probs, axis=0)
    test_targets = np.argmax(y_test_eval, axis=0)
    test_accuracy = np.mean(test_predictions == test_targets) * 100

    epoch_time = time.time() - start_time

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"  Average loss: {average_loss:.4f}")
    print(f"  Train accuracy: {train_accuracy:.2f}%")
    print(f"  Test accuracy ({test_eval_size} samples): {test_accuracy:.2f}%")
    print(f"  Time: {epoch_time:.1f}s")
    print("==============================")
