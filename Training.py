import numpy as np
from Loading import *

X_train, y_train_enc, X_test, y_test_enc = load_mnist_data(
    "train-images.idx3-ubyte",
    "train-labels.idx1-ubyte",
    "t10k-images.idx3-ubyte",
    "t10k-labels.idx1-ubyte",
)


# 1. Initialize weights and biases globally
W1 = np.random.randn(800, 784) * 0.01
b1 = np.zeros((800,))
W2 = np.random.randn(10, 800) * 0.01
b2 = np.zeros((10,))


num_epochs = 10
batch_size = 64


# 2. Training functions for loop
def ReLU(Z):
    return np.maximum(0, Z)


def ReLUPrime(Z):
    return (Z > 0).astype(float)


def softMax(a):
    a = a - np.max(a, axis=1, keepdims=True)
    exp_a = np.exp(a)
    return exp_a / np.sum(exp_a, axis=1, keepdims=True)


def hiddenLayer(X_batch, W1, b1):
    Z1 = np.dot(X_batch, np.transpose(W1)) + b1
    return ReLU(Z1)


def outputLayer(Z1, W2, b2):
    Z2 = np.dot(Z1, np.transpose(W2)) + b2
    return softMax(Z2)


def cross_entropy_loss(y_batch, Z2):
    Z2 = np.clip(Z2, 1e-15, 1)
    return -np.mean(np.sum(y_batch * np.log(Z2), axis=1))


def backProp(X_batch, y_batch, Z1, Z2, W2):
    # output layer loss
    delta2 = Z2 - y_batch
    W2loss = np.dot(np.transpose(delta2), Z1)
    B2loss = np.sum(delta2, axis=0, keepdims=True)

    # hidden layer loss
    delta1 = np.dot(delta2, W2) * ReLUPrime(Z1)
    W1loss = np.dot(np.transpose(delta1), X_batch)
    B1loss = np.sum(delta1, axis=0)

    return (W1loss, B1loss, W2loss, B2loss)


def update(W1, b1, W2, b2, W1loss, B1loss, W2loss, B2loss, learning_rate=0.01):
    learning_rate = 0.01
    W1 -= learning_rate * W1loss
    b1 -= learning_rate * B1loss.flatten()  # ensure shape
    W2 -= learning_rate * W2loss
    b2 -= learning_rate * B2loss.flatten()
    return W1, b1, W2, b2


# 3. Testing
def evaluate_accuracy(X, y_onehot, W1, b1, W2, b2):
    A1_test = hiddenLayer(X, W1, b1)  # (num_samples, 800)
    A2_test = outputLayer(A1_test, W2, b2)  # (num_samples, 10)
    predictions = np.argmax(A2_test, axis=1)
    true_labels = np.argmax(y_onehot, axis=1)
    return np.mean(predictions == true_labels)


# 4. Training loop

for epoch in range(num_epochs):

    print(f"Starting epoch {epoch+1}/{num_epochs}...")
    for X_batch, y_batch in create_batches(X_train, y_train_enc, batch_size):

        # X_batch shape => (64, 784)
        # y_batch shape => (64, 10)
        #
        # 1. Forward pass
        A1 = hiddenLayer(X_batch, W1, b1)
        A2 = outputLayer(A1, W2, b2)
        # 2. Compute loss
        L = cross_entropy_loss(y_batch, A2)
        # 3. Backprop (compute gradients)
        losses = backProp(X_batch, y_batch, A1, A2, W2)
        # 4. Update weights/biases
        W1, b1, W2, b2 = update(W1, b1, W2, b2, *losses)
        pass

    train_acc = evaluate_accuracy(X_train[:5000], y_train_enc[:5000], W1, b1, W2, b2)

    print(f"Epoch {epoch+1}/{num_epochs} finished!")


# 6. Final Evaluation on the FULL Test Set
test_acc = evaluate_accuracy(X_test, y_test_enc, W1, b1, W2, b2)
print(f"Test Accuracy on 10k images: {test_acc*100:.2f}%")

# 7. Save final weights
np.savez("trained_model.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("Model training complete and weights saved to trained_model.npz")
