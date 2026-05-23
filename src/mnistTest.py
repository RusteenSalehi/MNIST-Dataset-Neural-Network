from sklearn.datasets import load_digits
import numpy as np
from network import Network

digits = load_digits()
net = Network()

learning_rate = 0.00001
epochs = 100
batch_size = 32

for epoch in range(epochs):
    total_loss = 0
    indices = np.random.permutation(len(digits.images))

    for start in range(0, len(indices), batch_size):

        batch_indices = indices[start:start + batch_size]

        x_batch = []
        y_batch = []
        for i in batch_indices:

            image = digits.images[i]
            label = digits.target[i]

            x = image.reshape(64,1) / 16

            target = np.zeros((10, 1))
            target[label][0] = 1

            x_batch.append(x)
            y_batch.append(target)

        x_batch = np.hstack(x_batch)
        y_batch = np.hstack(y_batch)
        prediction = net.forward(x_batch)
        loss = net.cross_entropy_loss(prediction, y_batch)

        total_loss += loss
        net.backward(x_batch, y_batch, learning_rate)

    average_loss = total_loss / len(digits.images)

    # if epoch % 10 == 0:
    #     print(f"Epoch {epoch + 1}")
    #     print(f"Average Loss: {average_loss}")
    #
    # correct = 0
    # total = 10
    #
    # print("\nTesting Sample Predictions\n")
    #
    # for i in range(10):
    #     image = digits.images[i]
    #
    #     x = image.reshape(64, 1) / 16
    #
    #     prediction = net.forward(x)
    #     predicted_digit = np.argmax(prediction)
    #     actual_digit = digits.target[i]
    #
    #     if predicted_digit == actual_digit:
    #         correct += 1
    #
    #     print(f"Predicted: {predicted_digit}")
    #     print(f"Actual: {actual_digit}")
    #     print("-----------------------")
    #
    # accuracy = (correct / total) * 100
    #
    # print(f"Accuracy: {accuracy:.2f}%")
    # print("=================================")