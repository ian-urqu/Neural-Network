import numpy as np


def read_idx_images(filename):
    """
    Reads an unzipped MNIST 'images' file (e.g. 'train-images.idx3-ubyte')
    and returns a NumPy array of shape (num_images, 28, 28) with dtype=uint8.
    """
    with open(filename, "rb") as f:
        # Read first 4 bytes: magic number (not needed except for validation)
        magic = int.from_bytes(f.read(4), byteorder="big")
        # Read next 4 bytes: number of images
        num_images = int.from_bytes(f.read(4), byteorder="big")
        # Read next 4 bytes: rows
        num_rows = int.from_bytes(f.read(4), byteorder="big")
        # Read next 4 bytes: columns
        num_cols = int.from_bytes(f.read(4), byteorder="big")

        # Read the rest as 8-bit unsigned byte data
        data = f.read(num_images * num_rows * num_cols)
        arr = np.frombuffer(data, dtype=np.uint8)

        # Reshape to (num_images, 28, 28)
        arr = arr.reshape(num_images, num_rows, num_cols)
        return arr


def read_idx_labels(filename):
    """
    Reads an unzipped MNIST 'labels' file (e.g. 'train-labels.idx1-ubyte')
    and returns a NumPy array of shape (num_labels,) with dtype=uint8.
    """
    with open(filename, "rb") as f:
        # Read first 4 bytes: magic number (again, mostly for validation)
        magic = int.from_bytes(f.read(4), byteorder="big")
        # Read next 4 bytes: number of labels
        num_labels = int.from_bytes(f.read(4), byteorder="big")

        # Read the rest as 8-bit unsigned data
        data = f.read(num_labels)
        arr = np.frombuffer(data, dtype=np.uint8)
        return arr


def load_mnist_data(
    train_images_path="train-images.idx3-ubyte",
    train_labels_path="train-labels.idx1-ubyte",
    test_images_path="t10k-images.idx3-ubyte",
    test_labels_path="t10k-labels.idx1-ubyte",
):
    # Step 1: Read images and labels
    X_train_raw = read_idx_images(train_images_path)  # (60000, 28, 28)
    y_train_raw = read_idx_labels(train_labels_path)  # (60000,)
    X_test_raw = read_idx_images(test_images_path)  # (10000, 28, 28)
    y_test_raw = read_idx_labels(test_labels_path)  # (10000,)

    # Step 2: Flatten the images to (num_samples, 784)
    num_train = X_train_raw.shape[0]
    num_test = X_test_raw.shape[0]
    X_train = X_train_raw.reshape(num_train, 28 * 28)
    X_test = X_test_raw.reshape(num_test, 28 * 28)

    # Step 3: Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Step 4: One-hot encode the labels (digits 0–9 → vectors of length 10)
    y_train_encoded = one_hot_encode(y_train_raw, 10)
    y_test_encoded = one_hot_encode(y_test_raw, 10)

    return X_train, y_train_encoded, X_test, y_test_encoded


def one_hot_encode(labels, num_classes=10):
    """
    Converts label array of shape (num_samples,) to one-hot form (num_samples, num_classes).
    Example: label 3 → [0,0,0,1,0,0,0,0,0,0]
    """
    return np.eye(num_classes)[labels]


def create_batches(X, y, batch_size=64):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]
