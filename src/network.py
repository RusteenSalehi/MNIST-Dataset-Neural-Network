import numpy as np
from numpy.distutils.fcompiler import none


#Neural Network Class
class Network:
    def __init__(self):
        #Layer 1: 784 -> 100
        self.W1 = np.random.randn(50, 64) * np.sqrt(2/64)
        self.b1 = np.random.randn(50, 1)

        #Layer 2: 100 -> 50
        self.W2 = np.random.randn(20,50) * np.sqrt(2/50)
        self.b2 = np.random.randn(20, 1)

        #Layer 3: 50 -> 10
        self.W3 = np.random.randn(10, 20) * np.sqrt(2/20)
        self.b3 = np.random.randn(10, 1)

        #Initialize all values to none
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None
        self.Z3 = None
        self.A3 = None
    #Forward reasoning function based on weights and biases
    def forward(self, x):
        self.Z1 = np.matmul(self.W1, x) + self.b1
        self.A1 = self.ReLU(self.Z1)
        self.Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.A2 = self.ReLU(self.Z2)
        self.Z3 = np.matmul(self.W3, self.A2) + self.b3
        self.A3 = self.softmax(self.Z3) #Use softmax to turn output to probabilities
        return self.A3

    #Basic ReLU function to keep all values positive
    def ReLU(self, Z):
        return np.maximum(Z, 0)

    #Derivative ReLU for gradient descent (0 for negative numbers, 1 for positive numbers)
    def derivative_Relu(self, Z):
        return (Z > 0).astype(float)

    #Mean Squared Error loss function
    def mse_loss_function(self, prediction, target):
        return np.mean((prediction - target) ** 2)

    #Back propagation function
    def backward(self, x, y, lr):
        #Taking partial derivatives for each parameter
        d_z3 = 2 * (self.A3 - y)
        d_w3 = np.matmul(d_z3, self.A2.T)
        d_b3 = d_z3
        d_a2 = np.matmul(self.W3.T, d_z3)
        d_z2 = d_a2 * self.derivative_Relu(self.Z2)
        d_w2 = np.matmul(d_z2, self.A1.T)
        d_b2 = d_z2
        d_a1 = np.matmul(self.W2.T, d_z2)
        d_z1 = d_a1 * self.derivative_Relu(self.Z1)
        d_w1 = np.matmul(d_z1, x.T)
        d_b1 = d_z1
        #Updating parameters after gradient descent
        self.W1 = self.W1 - lr * d_w1
        self.b1 = self.b1 - lr * d_b1
        self.W2 = self.W2 - lr * d_w2
        self.b2 = self.b2 - lr * d_b2
        self.W3 = self.W3 - lr * d_w3
        self.b3 = self.b3 - lr * d_b3

    #Soft-max function to turn output layers to probability
    def softmax(self, z):
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)