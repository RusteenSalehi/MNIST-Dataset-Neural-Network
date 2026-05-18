from sklearn.datasets import load_digits
import numpy as np
from network import Network

digits = load_digits()
net = Network()
lr = 0.00001
epochs = 10

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(digits.images)):
        image = digits.images[i]
        label = digits.target[i]

        x = image.reshape(64,1) / 16

        target = np.zeros((10,1))
        target[label][0] = 1

        prediction = net.forward(x)

        loss = net.mse_loss_function(prediction, target)
        total_loss += loss

        net.backward(x, target, lr)

    average_loss = total_loss / len(digits.images)

    print(f"Epoch {epoch + 1}")
    print("Average Loss:", average_loss)

    test_image = digits.images[0].reshape(64, 1)

    test_prediction = net.forward(test_image)
    predicted_digit = np.argmax(test_prediction)
    print("Predicted Digits:", predicted_digit)
    print("Actual Digit:", digits.target[0])
    print("--------------------------")
print("\nTesting Network\n")
for i in range(10):
    image = digits.images[i]
    x = image.reshape(64,1)
    prediction = net.forward(x)
    predicted_digit = np.argmax(prediction)
    actual_digit = digits.target[i]
    print("Predicted:", predicted_digit)
    print("Actual:", actual_digit)
    print("--------------------------")