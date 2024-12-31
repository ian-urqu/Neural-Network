import numpy as np
import gzip

def load_images(filename):
    with gzip.open(filename, "rb") as file:
        file.read(16)
        buffer = file.read()
        data = np.frombuffer(buffer, dtype=np.uint8)

        data = data / 255.0

        num_images = len(data) // (28 * 28)
        return data.reshape(num_images, 28 * 28)
    

def load_labels(filename):
    with gzip.open(filename, "rb") as file:
        file.read(8)

        buffer = file.read()

        data = np.frombuffer(buffer, dtype=np.uint8)
        return data
    

