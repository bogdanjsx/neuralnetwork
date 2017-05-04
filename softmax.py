import numpy as np
from transfer_functions import softmax

from layer_interface import LayerInterface

class SoftmaxLayer(LayerInterface):

    def __init__(self):
        pass

    def forward(self, inputs):
        self.outputs = softmax(inputs)
        return self.outputs

    def backward(self, inputs, output_errors):
        Z = np.sum(np.multiply(output_errors, self.outputs))
        dX = np.multiply(self.outputs, output_errors - Z)
        return dX

    def to_string(self):
        return "[SoftMax]"