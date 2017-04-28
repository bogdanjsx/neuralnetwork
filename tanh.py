import numpy as np
from transfer_functions import hyperbolic_tangent

from layer_interface import LayerInterface

class TanhLayer(LayerInterface):

    def __init__(self):
        pass

    def forward(self, inputs):
        self.outputs = hyperbolic_tangent(inputs)
        return self.outputs

    def backward(self, inputs, output_errors):
        # return output_errors * hyperbolic_tangent(inputs, True)
        return np.multiply(1 - np.multiply(self.outputs, self.outputs), output_errors)

    def to_string(self):
        return "[TanH]"