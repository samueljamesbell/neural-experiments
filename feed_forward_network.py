"""A simple feedforward neural network with a fixed number of layers. """
import numpy as np

import activation
import cost
import iris_data_loader
import utils


_INPUT_LAYER_DIMENSIONS = 4
_HIDDEN_LAYER_DIMENSIONS = 3
_OUTPUT_LAYER_DIMENSIONS = 3

_TRAINING_EPOCHS = 25000
_LEARNING_RATE = 0.0001


class FeedForwardNet(object):

    # TODO: Have _feed_forward return a list of activation layer vectors and
    # weighted input layer vectors.
    # Backprop could take these and return a list of error layer vectors.
    # Gradient descent takes this, and actually updates weights and biases.§

    # TODO: Use softmax output layer and cross-entropy loss function.

    def __init__(self, number_training_examples):
        self.W_x_h = np.random.randn(_HIDDEN_LAYER_DIMENSIONS,
                                     _INPUT_LAYER_DIMENSIONS)
        self.W_h_o = np.random.randn(_OUTPUT_LAYER_DIMENSIONS,
                                     _HIDDEN_LAYER_DIMENSIONS)
        self.b_h = np.random.randn(_HIDDEN_LAYER_DIMENSIONS, 1)
        self.b_o = np.random.randn(_OUTPUT_LAYER_DIMENSIONS, 1)

        self.Z_h = np.zeros(
            [number_training_examples, _HIDDEN_LAYER_DIMENSIONS])
        self.Z_o = np.zeros(
            [number_training_examples, _OUTPUT_LAYER_DIMENSIONS])

        self.A_h = np.zeros(
            [number_training_examples, _HIDDEN_LAYER_DIMENSIONS])
        self.A_o = np.zeros(
            [number_training_examples, _OUTPUT_LAYER_DIMENSIONS])

    def train(self, X, Y, label_set):
        """Train given a set of data points, X, and gold labels, Y.

        label_set is the set of all possible gold labels.
        """
        Y_probabilities = utils.to_categorical_identity(Y, label_set)

        for i in range(0, _TRAINING_EPOCHS):
            self._forward_pass(X)
            # TODO: Restore softmax output layer.
            # softmax_outputs = utils.softmax(self.A_o)
            # cost = cost.quadratic_cost_prime(softmax_outputs, Y_probabilities)
            C = cost.quadratic_cost_prime(self.A_o, Y_probabilities)
            self._back_propagate(C, X)

        accuracy = utils.accuracy(utils.take_max(self.A_o), Y_probabilities)
        print('Training accuracy: {}'.format(accuracy))

    def test(self, X, Y, label_set):
        """Test infererence given  a set of data points, X, and gold labels, Y.

        label_set is the set of all possible gold labels.
        """
        Y_probabilities = utils.to_categorical_identity(Y, label_set)
        self._forward_pass(X)
        accuracy = utils.accuracy(utils.take_max(self.A_o), Y_probabilities)
        print('Test accuracy: {}'.format(accuracy))

    def _forward_pass(self, X):
        """Forward pass all points in X through the network.

        Stores both weighted inputs and activations for each layer.
        """
        self.Z_h = self.W_x_h.dot(X) + self.b_h
        self.A_h = activation.sigmoid(self.Z_h)

        self.Z_o = self.W_h_o.dot(self.A_h) + self.b_o
        self.A_o = activation.sigmoid(self.Z_o)

    def _back_propagate(self, cost, X):
        """Update weights and biases according to the cost.

        Uses backpropagation to apportion cost to each parameter.

        Currently also performs batch (that's right, the whole batch) gradient
        descent to actually update the parameters. We should probably just
        calculate the error of each layer here.
        """
        output_layer_error = np.multiply(
            cost, activation.sigmoid_prime(self.Z_o))
        hidden_layer_error = np.multiply(self.W_h_o.T.dot(
            output_layer_error), activation.sigmoid_prime(self.Z_h))

        output_layer_bias_gradients = output_layer_error
        hidden_layer_bias_gradients = hidden_layer_error

        output_layer_weight_gradients = output_layer_error.dot(self.A_h.T)
        hidden_layer_weight_gradients = hidden_layer_error.dot(X.T)

        # TODO: This is gradient descent - should probably be its own method

        self.b_o = self.b_o - (np.mean(output_layer_bias_gradients,
                                       axis=1).reshape(self.b_o.shape[0], 1) *
                               _LEARNING_RATE)
        self.b_h = self.b_h - (np.mean(hidden_layer_bias_gradients,
                                       axis=1).reshape(self.b_h.shape[0], 1) *
                               _LEARNING_RATE)

        self.W_h_o = self.W_h_o - \
            (output_layer_weight_gradients * _LEARNING_RATE)
        self.W_x_h = self.W_x_h - \
            (hidden_layer_weight_gradients * _LEARNING_RATE)


if __name__ == '__main__':
    X_train, Y_train = iris_data_loader.training_data()
    X_test, Y_test = iris_data_loader.test_data()
    n = FeedForwardNet(X_train.shape[0])
    n.train(X_train, Y_train, iris_data_loader.LABELS)
    n.test(X_test, Y_test, iris_data_loader.LABELS)
