import numpy as np 

def relu(x):
    return np.maximum(x, 0)

def drelu(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def cross_entropy(y, y_hat):
    return -np.sum(y * np.log(y_hat), axis=1, keepdims=True)

def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y]