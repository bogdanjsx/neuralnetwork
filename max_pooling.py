import numpy as np

from layer_interface import LayerInterface

class MaxPoolingLayer(LayerInterface):

    def __init__(self, stride):
        # Dimensions: stride
        self.stride = stride

        # indexes of max activations
        self.switches = {}

    def forward(self, inputs):
        (d, w, h) = inputs.shape
        s = self.stride

        self.outputs = inputs.reshape(d, w // s, s, h // s, s).max(axis=(2, 4))
        return self.outputs

        # N,h,w = x.shape
        # inputs = inputs.reshape(d,w/2,2,h/2,2).swapaxes(2,3).reshape(d,w/2,h/2,4)
        # return np.amax(inputs,axis=3)

        # result = np.zeros((d, w // s, h // s))

        # for i in range(d):
        #     for j in range(w // s):
        #         for k in range(h // s):
        #             result[i, j, k] = np.amax(inputs[i, j*s : (j+1)*s, k*s : (k+1)*s])

        # self.outputs = result
        # return result

    def backward(self, inputs, output_errors):
        (d, w, h) = inputs.shape

        s = self.stride
        result = np.zeros(inputs.shape)

        for i in range(d):
            for j in range(w):
                for k in range(h):
                    if inputs[i, j, k] == output_errors[i, j // 2, k // 2]:
                        result[i, j, k] = inputs[i, j, k]

        return result

    def to_string(self):
        return "[MP (%s x %s)]" % (self.stride, self.stride)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

from util import close_enough

def test_max_pooling_layer():

    l = MaxPoolingLayer(2)

    x = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]],
                  [[9, 10, 11, 12], [13, 14, 15, 16]],
                  [[17, 18, 19, 20], [21, 22, 23, 24]]])

    print("Testing forward computation...")
    output = l.forward(x)
    target = np.array([[[6, 8]],
                       [[14, 16]],
                       [[22, 24]]])
    assert (output.shape == target.shape), "Wrong output size"
    assert close_enough(output, target), "Wrong values in layer ouput"
    print("Forward computation implemented ok!")


    output_err = output

    print("Testing backward computation...")

    g = l.backward(x, output_err)
    print(g)


    print("Testing gradients")
    in_target = np.array([[[0, 0, 0, 0], [0, 6, 0, 8]],
                          [[0, 0, 0, 0], [0, 14, 0, 16]],
                          [[0, 0, 0, 0], [0, 22, 0, 24]]])

    assert (g.shape == in_target.shape), "Wrong size"
    assert close_enough(g, in_target), "Wrong values in gradients"
    print("     OK")

    print("Backward computation implemented ok!")


if __name__ == "__main__":
    test_max_pooling_layer()
