from sklearn.datasets import load_digits
import numpy as np
from network import Network

digits = load_digits()
net = Network()
lr = 0.00001
epochs = 10

for epoch in range(epochs):
    totalLoss = 0
    for i in range(len(digits.images)):
        image = digits.images[i]
        label = digits.target[i]

        x = image.reshape(64,1)

        target = np.zeros((10,1))
        target[label][0] = 1

        prediction = net.forward(x)

        loss = net.mseLossFunction(prediction, target)
        totalLoss += loss

        net.backward(x, target, lr)

    averageLoss = totalLoss / len(digits.images)

    print(f"Epoch {epoch + 1}")
    print("Average Loss:", averageLoss)

    testImage = digits.images[0].reshape(64,1)

    testPrediction = net.forward(testImage)
    predictedDigit = np.argmax(testPrediction)
    print("Predicted Digits:", predictedDigit)
    print("Actual Digit:", digits.target[0])
    print("--------------------------")
print("\nTesting Network\n")
for i in range(10):
    image = digits.images[i]
    x = image.reshape(64,1)
    prediction = net.forward(x)
    predictedDigit = np.argmax(prediction)
    actualDigit = digits.target[i]
    print("Predicted:", predictedDigit)
    print("Actual:", actualDigit)
    print("--------------------------")