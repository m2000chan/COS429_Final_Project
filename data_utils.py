import os, pickle
import numpy as np
import random
from scipy import ndimage

# CIFAR-10: mean (0.4914, 0.4822, 0.4465), std (0.247, 0.243, 0.261)

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, "rb") as f:
        datadict = pickle.load(f, encoding="latin1")
        X = datadict["data"]; Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y
    
def apply_random_rotations(X, y):
    """
    Apply random rotations to the test dataset.
    """
    num_samples = X.shape[-1]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Define the portions for each rotation type
    portion = num_samples // 4
    left_indices = indices[:portion]
    right_indices = indices[portion: 2 * portion]
    flip_indices = indices[2 * portion: 3 * portion]

    # Apply the rotations
    X_rotated = np.copy(X)
    X_rotated[..., left_indices] = np.rot90(X[..., left_indices], k=1, axes=(0, 1))
    X_rotated[..., right_indices] = np.rot90(X[..., right_indices], k=-1, axes=(0, 1))
    X_rotated[..., flip_indices] = np.rot90(X[..., flip_indices], k=2, axes=(0, 1))

    return X_rotated, y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []; ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X); ys.append(Y)
    Xtr = np.concatenate(xs); Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(cifar10_dir='./cifar-10-batches-py', num_training=50000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for training.
    """
    
    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
    try:
        del X_train, y_train
        del X_test, y_test
        print('Clear previously loaded data.')
    except:
        pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Normalize
    X_train  = X_train / 255.0
    X_test = X_test / 255.0
    means = np.array([0.4914, 0.4822, 0.4465])
    stds = np.array([0.247, 0.243, 0.261])
    X_train = (X_train - means) / stds
    X_test = (X_test - means) / stds

    # H x W x C x Batch_Size
    X_train = np.transpose(X_train, (1,2,3,0))
    X_test = np.transpose(X_test, (1,2,3,0))

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    # Apply random rotations to the test dataset
    X_test, y_test = apply_random_rotations(X_test, y_test)

    # # Print shapes
    # print('Train data shape: ', X_train.shape)
    # print('Train labels shape: ', y_train.shape)
    # print('Test data shape: ', X_test.shape)
    # print('Test labels shape: ', y_test.shape)
    
    return X_train, y_train, X_test, y_test
