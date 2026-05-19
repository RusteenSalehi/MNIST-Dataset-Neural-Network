from sklearn.datasets import load_digits
import numpy as np
from network import Network

digits = load_digits()
net = Network()
lr = 0.00001
epochs = 100

for epoch in range(epochs):
    indices = np.random.permutation(len(digits.images))
    total_loss = 0
    for i in indices:
        image = digits.images[i]
        label = digits.target[i]

        x = image.reshape(64,1) / 16

        target = np.zeros((10,1))
        target[label][0] = 1

        prediction = net.forward(x)

        loss = net.cross_entropy_loss(prediction, target)
        total_loss += loss

        net.backward(x, target, lr)

    average_loss = total_loss / len(digits.images)

    if epoch % 10 == 0:
        print(f"Epoch {epoch + 1}")
        print("Average Loss:", average_loss)

print("\nTesting Network\n")
correct = 0
for i in range(10):
    image = digits.images[i]
    x = image.reshape(64,1) / 16
    prediction = net.forward(x)
    predicted_digit = np.argmax(prediction)
    actual_digit = digits.target[i]
    print("Predicted:", predicted_digit)
    print("Actual:", actual_digit)
    print("--------------------------")
    if actual_digit == predicted_digit:
        correct += 1
accuracy = (correct / 10) * 100
print("This model ran with ", accuracy, "% accuracy")