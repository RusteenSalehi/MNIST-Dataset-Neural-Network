import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
image = x_train[0]
print(image.shape)

kernal = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

print(kernal.shape)