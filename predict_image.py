import numpy as np
import matplotlib.pyplot as plt
from Loading import *


weights = np.load("trained_model.npz")
W1 = weights["W1"]
b1 = weights["b1"]
W2 = weights["W2"]
b2 = weights["b2"]


def ReLU(Z):
    return np.maximum(0, Z)


def softMax(a):
    a = a - np.max(a, axis=1, keepdims=True)
    exp_a = np.exp(a)
    return exp_a / np.sum(exp_a, axis=1, keepdims=True)


def hiddenLayer(X, W1, b1):
    Z1 = np.dot(X, W1.T) + b1
    return ReLU(Z1)


def outputLayer(Z1, W2, b2):
    Z2 = np.dot(Z1, W2.T) + b2
    return softMax(Z2)


X_train, y_train_enc, X_test, y_test_encoded = load_mnist_data()

idx = np.random.randint(0, X_train.shape[0])

random_img_2d = X_train[idx].reshape(28, 28)

true_label = np.argmax(y_train_enc[idx])


plt.imshow(random_img_2d, cmap="gray")
plt.title(f"True Label: {true_label}")
plt.axis("off")
plt.show()

random_img_1d = X_train[idx].reshape(1, 784)
A1 = hiddenLayer(random_img_1d, W1, b1)
A2 = outputLayer(A1, W2, b2)
predicted_label = np.argmax(A2)

print(f"Predicted Label: {predicted_label}")
