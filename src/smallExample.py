#Test file on smaller datasets to test theory and math
import numpy as np

#input
x=np.array([
    [2],
    [3]
])

#Correct answer
y=1

#initial weights and biases
W=np.array([[1.0, 2.0]])
b=1

#Learning rate
lr = 0.01

#Number of iterations
epochs = 100
prediction = 0
loss = 0

for epoch in range(epochs):
    #Forward
    prediction = np.matmul(W, x) + b
    loss = (prediction - y) ** 2

    #Back prop
    dL_dPrediction = 2 * (prediction - y)
    dW = dL_dPrediction * x.T
    db = dL_dPrediction

    #Gradient descent
    W = W - lr * dW
    b = b - lr * db

    #Print every tenth time
    if epoch % 10 == 0:
        print(f"Epoch {epoch}")
        print("Prediction:", prediction)
        print("Loss:", loss)
        print("Weights:", W)
        print("Bias:", b)
        print("---------------")

print("Final Results")
print("Prediction:", prediction)
print("Loss:", loss)
print("Weights:", W)
print("Bias:", b)