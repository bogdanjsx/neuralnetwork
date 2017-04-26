# Tudor Berariu, 2015
import numpy as np                                  # Needed to work with arrays
from argparse import ArgumentParser

import matplotlib
matplotlib.use('TkAgg')
import pylab

from data_loader import load_mnist
from feed_forward import FeedForward
from transfer_functions import identity, logistic, hyperbolic_tangent
from layer import Layer
from convolutional import ConvolutionalLayer
from max_pooling import MaxPoolingLayer
from relu import ReluLayer
from linearize import LinearizeLayer, LinearizeLayerReverse

def eval_nn(nn, imgs, labels, maximum = 0):
    # Compute the confusion matrix
    confusion_matrix = np.zeros((10, 10))
    correct_no = 0
    how_many = imgs.shape[0] if maximum == 0 else maximum
    for i in range(imgs.shape[0])[:how_many]:
        y = np.argmax(nn.forward(imgs[i]))
        t = labels[i]
        if y == t:
            correct_no += 1
        confusion_matrix[y][t] += 1

    return float(correct_no) / float(how_many), confusion_matrix / float(how_many)

def train_nn(nn, data, args):
    pylab.ion()
    cnt = 0
    print(data)
    for i in np.random.permutation(data["train_no"]):

        cnt += 1

        inputs = data["train_imgs"][i]
        label = data["train_labels"][i]
        targets = np.zeros((10, 1))
        targets[label] = 1
        outputs = nn.forward(inputs)
        errors = outputs - targets
        nn.backward(inputs, errors)
        nn.update_parameters(args.learning_rate)

        # Evaluate the network
        if cnt % args.eval_every == 0:
            test_acc, test_cm = \
                eval_nn(nn, data["test_imgs"], data["test_labels"])
            train_acc, train_cm = \
                eval_nn(nn, data["train_imgs"], data["train_labels"], 5000)
            print("Train acc: %2.6f ; Test acc: %2.6f" % (train_acc, test_acc))
            pylab.imshow(test_cm)
            pylab.draw()

            matplotlib.pyplot.pause(0.001)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", type = float, default = 0.001,
                        help="Learning rate")
    parser.add_argument("--eval_every", type = int, default = 200,
                        help="Learning rate")
    args = parser.parse_args()

    mnist = load_mnist()
    input_size = mnist["train_imgs"][0].size

    # TODO 5
    nn = FeedForward([Layer(input_size, 300, logistic), Layer(300, 10, identity)])
    # nn = FeedForward([LinearizeLayer(), Layer(300, )])
    # nn = FeedForward([LinearizeLayerReverse(1, 28, 28), ConvolutionalLayer(1, 28, 28, 16, 5, 1), MaxPoolingLayer(2), ReluLayer(), ConvolutionalLayer(16, 12, 12, 16, 5, 1), MaxPoolingLayer(2), ReluLayer(), LinearizeLayer(16, 4, 4), Layer(256, 10, identity)])
    # print(nn.to_string())
    print(mnist)
    # train_nn(nn, mnist, args)