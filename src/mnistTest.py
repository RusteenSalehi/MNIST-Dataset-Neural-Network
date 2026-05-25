# from keras.datasets import mnist
# import numpy as np
# from network import Network
#
# #Load the real MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# net = Network()
#
# learning_rate = 0.0001
# epochs = 50
# batch_size = 32
#
# for epoch in range(epochs):
#     total_loss = 0
#     indices = np.random.permutation(len(x_train))
#
#     for start in range(0, len(indices), batch_size):
#         batch_indices = indices[start:start + batch_size]
#
#         x_batch = []
#         y_batch = []
#
#         for i in batch_indices:
#             image = x_train[i]
#             label = y_train[i]
#
#             x = image.reshape(784,1) / 255.0
#
#             target = np.zeros((10, 1))
#             target[label][0] = 1
#
#             x_batch.append(x)
#             y_batch.append(target)
#
#         x_batch = np.hstack(x_batch)
#         y_batch = np.hstack(y_batch)
#
#         prediction = net.forward(x_batch)
#         loss = net.cross_entropy_loss(prediction, y_batch)
#         total_loss += loss
#
#         net.backward(x_batch, y_batch, learning_rate)
#
#     average_loss = total_loss / len(x_train)
#
#     if epoch % 1 == 0:
#         print(f"Epoch {epoch + 1}")
#         print(f"Average Loss: {average_loss}")
#
#         correct = 0
#         total = 1000
#
#         for i in range(total):
#             image = x_test[i]
#
#             x = image.reshape(784, 1) / 255.0
#
#             prediction = net.forward(x)
#             predicted_digit = np.argmax(prediction)
#
#             actual_digit = y_test[i]
#
#             if predicted_digit == actual_digit:
#                 correct += 1
#
#         accuracy = (correct / total) * 100
#         print(f"Accuracy: {accuracy:.2f}%")
#         print("==============================")