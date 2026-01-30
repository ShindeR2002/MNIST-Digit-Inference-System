import kagglehub
import os
import numpy as np

def load_mnist_data():
    # 1. Download dataset via kagglehub
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    
    # 2. Define helper to read binary files
    def read_idx(filename):
        with open(filename, 'rb') as f:
            return np.frombuffer(f.read(), dtype=np.uint8, offset=16 if 'images' in filename else 8)

    # 3. Load and Normalize
    X_train = read_idx(os.path.join(path, "train-images-idx3-ubyte")).reshape(-1, 784) / 255.0
    y_train = read_idx(os.path.join(path, "train-labels-idx1-ubyte"))
    X_test = read_idx(os.path.join(path, "t10k-images-idx3-ubyte")).reshape(-1, 784) / 255.0
    y_test = read_idx(os.path.join(path, "t10k-labels-idx1-ubyte"))

    return X_train, y_train, X_test, y_test