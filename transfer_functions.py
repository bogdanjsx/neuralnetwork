import numpy as np
np.set_printoptions(threshold=np.nan)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# -- Transfer functions
#
# Useful numpy functions and constants:
#  - np.e : the "e" constant (Euler's number)

def identity(x, derivate = False):
    return x if not derivate else np.ones(x.shape)

def logistic(x, derivate = False):
    return 1 / (1 + np.e ** (-x)) if not derivate else x * (1 - x)

def hyperbolic_tangent(x, derivate = False):
    return np.tanh(x) if not derivate else 1 - x ** 2

def hyperbolic_tangent2(x, derivate = False):
    return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1) if not derivate else 1 - x ** 2  

def mx(x):
    return max(0, x)

def mx2(x):
    return 1 if x > 0 else 0

def relu(x, derivate = False):
    if not derivate:
        return np.vectorize(mx, otypes=[np.float])(x)
    return np.vectorize(mx2, otypes=[np.float])(x)

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()




