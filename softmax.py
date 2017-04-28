import numpy as np
from transfer_functions import softmax
from time import sleep

from layer_interface import LayerInterface

class SoftmaxLayer(LayerInterface):

    def __init__(self):
        pass

    def forward(self, inputs):
        self.outputs = softmax(inputs)
        # print("softmax forward")
        # print(self.outputs)
        # sleep(1)
        return self.outputs

    def backward(self, inputs, output_errors):
        # return inputs * (output_errors - inputs * output_errors)
        Z  = np.sum(np.multiply(output_errors, inputs))
        deltax = np.multiply(self.outputs, output_errors - Z)
        return deltax

    def to_string(self):
        return "[SoftMax]"