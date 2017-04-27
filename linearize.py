import numpy as np

from layer_interface import LayerInterface

class LinearizeLayer(LayerInterface):

    def __init__(self, depth, height, width):
        # Dimensions: depth, height, width
        self.depth = depth
        self.height = height
        self.width = width

    def forward(self, inputs):
        assert(inputs.shape == (self.depth, self.height, self.width))

        self.outputs = np.reshape(inputs, (self.depth * self.height * self.width, 1))
        return self.outputs

    def backward(self, inputs, output_errors):
        assert(output_errors.shape == (self.depth * self.height * self.width, 1))

        return np.reshape(output_errors, (self.depth, self.height, self.width))

    def to_string(self):
        return "[Lin ((%s, %s, %s) -> %s)]" % (self.depth, self.height, self.width, self.depth * self.height * self.width)


class LinearizeLayerReverse(LayerInterface):

    def __init__(self, depth, height, width):
        self.depth = depth
        self.height = height
        self.width = width

    def forward(self, inputs):
        assert(inputs.shape == (self.depth * self.height * self.width, 1))

        self.outputs = np.reshape(inputs, (self.depth, self.height, self.width))
        return self.outputs

    def backward(self, inputs, output_errors):
        assert(output_errors.shape == (self.depth, self.height, self.width))

        return np.matrix(output_errors.flatten()).T

    def to_string(self):
        return "[Lin (%s -> (%s, %s, %s))]" % (self.depth * self.height * self.width, self.depth, self.height, self.width)

