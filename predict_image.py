import numpy as np
from PIL import Image

# 1. Load saved weights
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


def load_and_preprocess_image(file_path):
    img = Image.open(file_path).convert("L")  # grayscale
    img = img.resize((28, 28))
    arr = np.array(img) / 255.0  # normalize
    arr = arr.flatten().reshape(1, 784)  # (1, 784)
    return arr


def predict_digit(file_path):
    X = load_and_preprocess_image(file_path)
    A1 = hiddenLayer(X, W1, b1)
    A2 = outputLayer(A1, W2, b2)
    return np.argmax(A2, axis=1)[0]


# Example usage
if __name__ == "__main__":
    digit_image = "test-digit-1.png"
    pred = predict_digit(digit_image)
    print(f"Predicted digit for {digit_image} is: {pred}")
