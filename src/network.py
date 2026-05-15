import numpy as np

#Neural Network Class
class Network:
    def __init__(self):
        #Layer 1: 784 -> 100
        self.W1 = np.random.randn(50, 64)
        self.b1 = np.random.randn(50, 1)

        #Layer 2: 100 -> 50
        self.W2 = np.random.randn(20,50)
        self.b2 = np.random.randn(20, 1)

        #Layer 3: 50 -> 10
        self.W3 = np.random.randn(10, 20)
        self.b3 = np.random.randn(10, 1)

    #Forward reasoning function based on weights and biases
    def forward(self, x):
        self.Z1 = np.matmul(self.W1, x) + self.b1
        self.A1 = self.ReLU(self.Z1)
        self.Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.A2 = self.ReLU(self.Z2)
        self.Z3 = np.matmul(self.W3, self.A2) + self.b3
        self.A3 = self.ReLU(self.Z3)
        return self.A3

    #Basic ReLU function to keep all values positive
    def ReLU(self, Z):
        for i in range(len(Z)):
            Z[i] = max(Z[i], 0)
        return Z

    #Derivative ReLU for gradient descent
    def derivativeReLU(self, Z):
        return (Z > 0).astype(float)

    #Mean Squared Error loss function
    def mseLossFunction(self, prediction, target):
        return np.mean((prediction - target) ** 2)

    #Back propagation function
    def backward(self, x, y, lr):
        #Taking partial derivatives for each parameter
        dA3 = 2 * (self.A3 - y)
        dZ3 = dA3 * self.derivativeReLU(self.Z3)
        dW3 = np.matmul(dZ3, self.A2.T)
        dB3 = dZ3
        dA2 = np.matmul(self.W3.T, dZ3)
        dZ2 = dA2 * self.derivativeReLU(self.Z2)
        dW2 = np.matmul(dZ2, self.A1.T)
        dB2 = dZ2
        dA1 = np.matmul(self.W2.T, dZ2)
        dZ1 = dA1 * self.derivativeReLU(self.Z1)
        dW1 = np.matmul(dZ1, x.T)
        dB1 = dZ1
        #Updating parameters after gradient descent
        self.W1 = self.W1 - lr * dW1
        self.b1 = self.b1 - lr * dB1
        self.W2 = self.W2 - lr * dW2
        self.b2 = self.b2 - lr * dB2
        self.W3 = self.W3 - lr * dW3
        self.b3 = self.b3 - lr * dB3