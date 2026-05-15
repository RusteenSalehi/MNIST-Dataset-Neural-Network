from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
print("Images shape:", digits.iamges.shape)
print("Labels shape:", digits.target.shape)

image = digits.images[0]
label = digits.target[0]

print("Label:", label)

print("Original image matrix")
print(image)

print("Original shape", image.shape)

flattened = image.reshape(64,1)

print("Flattened vector:")
print(flattened)

print("Flattened shape:", flattened.shape)

plt.imshow(image, cmap="gray")
plt.title(f"label: {label}")
plt.show()