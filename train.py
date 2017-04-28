import numpy as np                              # Needed to work with arrays
from argparse import ArgumentParser

from prepare_data import load_data

from feed_forward import FeedForward # The neural network
from transfer_functions import identity, logistic, hyperbolic_tangent

from layer import Layer # Fully connected
from tanh import TanhLayer
from softmax import SoftmaxLayer
from linearize import LinearizeLayer, LinearizeLayerReverse


from convolutional import ConvolutionalLayer
from max_pooling import MaxPoolingLayer
from relu import ReluLayer


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
        confusion_matrix[int(y)][int(t)] += 1

    return float(correct_no) / float(how_many), confusion_matrix / float(how_many)

def train_nn(nn, data, args):
    cnt = 0
    for i in np.random.permutation(data["train_no"]):

        cnt += 1

        inputs = data["train_imgs"][i]
        label = data["train_labels"][i]
        targets = np.zeros((10, 1))
        targets[int(label)] = 1
        outputs = nn.forward(inputs)

        errors = outputs - targets

        nn.backward(inputs, errors)
        nn.update_parameters(args.learning_rate)

        # Evaluate the network
        if cnt % args.eval_every == 0:
            nn.zero_gradients()
            test_acc, test_cm = \
                eval_nn(nn, data["test_imgs"], data["test_labels"])
            train_acc, train_cm = \
                eval_nn(nn, data["train_imgs"], data["train_labels"], 1000)
            print("Train acc: %2.6f ; Test acc: %2.6f" % (train_acc, test_acc))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", type = float, default = 0.001,
                        help="Learning rate")
    parser.add_argument("--eval_every", type = int, default = 200,
                        help="How often to evaluate")
    args = parser.parse_args()

    cifar = load_data()

    nn = FeedForward([LinearizeLayer(3, 32, 32), Layer(3 * 32 * 32, 300, logistic), TanhLayer(), Layer(300, 10, logistic), SoftmaxLayer()])
    # nn = FeedForward([LinearizeLayer(3, 32, 32), Layer(3 * 32 * 32, 10, identity), SoftmaxLayer()])

    # nn = FeedForward([ConvolutionalLayer(3, 32, 32, 5, 4, 2), ReluLayer(), LinearizeLayer(5, 15, 15), Layer(5 * 15 * 15, 10, identity)])

    print(nn.to_string())
    train_nn(nn, cifar, args)
