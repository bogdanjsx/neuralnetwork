import numpy as np

from layer_interface import LayerInterface
from utils import *

class ConvolutionalLayer(LayerInterface):

    def __init__(self, inputs_depth, inputs_height, inputs_width, outputs_depth, k, stride, use_momentum = False):
        # Number of inputs, number of outputs, filter size, stride

        self.inputs_depth = inputs_depth
        self.inputs_height = inputs_height
        self.inputs_width = inputs_width

        self.k = k
        self.stride = stride

        self.outputs_depth = outputs_depth
        self.outputs_height = int((self.inputs_height - self.k) / self.stride + 1)
        self.outputs_width = int((self.inputs_width - self.k) / self.stride + 1)

        # Layer's parameters
        self.weights = np.random.normal(
            0,
            np.sqrt(2.0 / float(self.outputs_depth + self.inputs_depth + self.k + self.k)),
            (self.outputs_depth, self.inputs_depth, self.k, self.k)
        )
        self.biases = np.random.normal(
            0,
            np.sqrt(2.0 / float(self.outputs_depth + 1)),
            (self.outputs_depth, 1)
        )

        # computed values
        self.outputs = np.zeros((self.outputs_depth, self.outputs_height, self.outputs_width))

        # Gradients
        self.g_weights = np.zeros(self.weights.shape)
        self.g_biases = np.zeros(self.biases.shape)

         # Momentum
        self.use_momentum = use_momentum
        self.v = np.zeros(self.weights.shape)
        self.mu = 0.9

    def forward(self, inputs):
        assert(inputs.shape == (self.inputs_depth, self.inputs_height, self.inputs_width))
        k = self.k

        # im2col indexes
        X_col = im2col(inputs, k, self.stride).T

        # reshaped weights
        W_row = self.weights.reshape((self.outputs_depth, k * k * self.inputs_depth))

        # Actual im2col values
        X_val = np.take(inputs, X_col)

        # Dot product of weights and actual columns
        res = np.dot(W_row, X_val) + self.biases
        
        self.outputs = res.reshape((self.outputs_depth, self.outputs_height, self.outputs_width))

        self.cache = X_col.T, X_val.T 

        return self.outputs

    def backward(self, inputs, output_errors):
        assert(output_errors.shape == (self.outputs_depth, self.outputs_height, self.outputs_width))

        (d, w, h) = inputs.shape
        (do, wo, ho) = output_errors.shape

        # Restore cache
        X_col, X_val = self.cache

        # compute the gradients w.r.t. the bias terms (self.g_biases)
        self.g_biases += output_errors.sum(axis=(1,2)).reshape(do, 1)

        # compute the gradients w.r.t. the weights (self.g_weights)
        dout_reshaped = output_errors.reshape(self.outputs_depth, -1)
        W_grad = np.dot(dout_reshaped, X_val)
        self.g_weights = W_grad.reshape(self.weights.shape)

        # compute and return the gradients w.r.t the inputs of this layer
        W_reshape = self.weights.reshape(self.outputs_depth, -1)

        dx_col = np.dot(W_reshape.T, dout_reshaped)
        dx = col2im(X_col, dx_col.T, inputs.shape)

        return dx

    def zero_gradients(self):
        self.g_biases = np.zeros(self.g_biases.shape)
        self.g_weights = np.zeros(self.g_weights.shape)

    def update_parameters(self, learning_rate):
        self.biases -= self.g_biases * learning_rate

        if self.use_momentum:
            self.v = self.mu * self.v - learning_rate * self.g_weights # integrate velocity
            self.weights += self.v
        else:
            self.weights -= self.g_weights * learning_rate

    def to_string(self):
        return "[C ((%s, %s, %s) -> (%s, %s) -> (%s, %s, %s)]" % (self.inputs_depth, self.inputs_height, self.inputs_width, self.k, self.stride, self.outputs_depth, self.outputs_height, self.outputs_width)
