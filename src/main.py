import numpy as np
import matplotlib
from network import Network #Importing neural network class

#Creating Neural Network
net = Network()

#Test code
x = np.random.randn(784, 1)
output = net.forward(x)
print(output.shape)