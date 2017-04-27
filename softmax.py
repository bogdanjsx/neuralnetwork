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
        return inputs * (output_errors - inputs * output_errors)

    def to_string(self):
        return "[SoftMax]"