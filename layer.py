# Tudor Berariu, 2016

import numpy as np

from layer_interface import LayerInterface
from time import sleep

class Layer(LayerInterface):

    def __init__(self, inputs_no, outputs_no, transfer_function):
        # Number of inputs, number of outputs, and the transfer function
        self.inputs_no = inputs_no
        self.outputs_no = outputs_no
        self.f = transfer_function

        # Layer's parameters - Xavier method
        self.weights = np.random.normal(
            0,
            np.sqrt(2.0 / float(self.outputs_no + self.inputs_no)),
            (self.outputs_no, self.inputs_no)
        )
        self.biases = np.random.normal(
            0,
            np.sqrt(2.0 / float(self.outputs_no + self.inputs_no)),
            (self.outputs_no, 1)
        )

        # Computed values
        self.a = np.zeros((self.outputs_no, 1))
        self.outputs = np.zeros((self.outputs_no, 1))

        # Gradients
        self.g_weights = np.zeros((self.outputs_no, self.inputs_no))
        self.g_biases = np.zeros((self.outputs_no, 1))


    def forward(self, inputs):
        assert(inputs.shape == (self.inputs_no, 1))

        self.a = np.dot(self.weights, inputs) + self.biases
        self.outputs = self.f(self.a)

        return self.outputs

    def backward(self, inputs, output_errors):
        assert(output_errors.shape == (self.outputs_no, 1))

        z = inputs
        fd = self.f(z, True)
        delta = np.dot(self.weights.T, output_errors) * fd

        self.g_biases = output_errors

        # Compute the gradients w.r.t. the weights (self.g_weights)
        self.g_weights = np.dot(inputs, output_errors.T).T

        # Compute and return the gradients w.r.t the inputs of this layer
        return delta

    def zero_gradients(self):
        self.g_biases = np.zeros(self.g_biases.shape)
        self.g_weights = np.zeros(self.g_weights.shape)

    def update_parameters(self, learning_rate):
        self.biases -= self.g_biases * learning_rate
        self.weights -= self.g_weights * learning_rate

    def to_string(self):
        return "[FC (%s -> %s)]" % (self.inputs_no, self.outputs_no)
