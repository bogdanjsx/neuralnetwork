import numpy as np
np.set_printoptions(threshold=np.nan)

# -- Transfer functions

def identity(x, derivate = False):
    return x if not derivate else np.ones(x.shape)

def logistic(x, derivate = False):
    return 1 / (1 + np.e ** (-x)) if not derivate else x * (1 - x)

def hyperbolic_tangent(x, derivate = False):
    return np.tanh(x) if not derivate else 1 - x ** 2

def relu(x, derivate = False):
    return np.maximum(x, 0) if not derivate else 1 * (x > 0)

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()
