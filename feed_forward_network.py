"""A simple feedforward neural network with a fixed number of layers.

Trained on the Fisher's Iris data set.
"""

import numpy as np

import iris_data_loader


_INPUT_LAYER_DIMENSIONS = 4
_HIDDEN_LAYER_DIMENSIONS = 3
_OUTPUT_LAYER_DIMENSIONS = 3

_TRAINING_EPOCHS = 50000
_LEARNING_RATE = 0.0001


def _take_max(M):
    """Given a matrix M, find the max values in each row.

    Returns a 1 in the column with the max value; 0 otherwise.
    """
    # TODO: This is very Iris-specific
    M_max = np.argmax(M, axis=0)

    N = np.zeros_like(M)
    N[0, :] = M_max == 0
    N[1, :] = M_max == 1
    N[2, :] = M_max == 2
    return N


def _to_categorical_identity(Y):
    """Return a sparse representation of a set of labels.

    For a given label y, return a vector with all probability density assigned
    to the yth index of the vector.

    e.g.  y = 2 yields [0, 0, 1]
    """
    # TODO: This is very Iris-specific
    Y_probabilities = np.zeros([Y.shape[0], len(np.unique(Y))])
    Y_probabilities[:, 0] = Y == 0
    Y_probabilities[:, 1] = Y == 1
    Y_probabilities[:, 2] = Y == 2
    return Y_probabilities.transpose()


def _softmax(M):
    """Compute Softmax of a given matrix, M."""
    M_exponents = np.exp(M)
    denominators = np.sum(M_exponents, axis=0)
    return np.divide(M_exponents, denominators)


def _activation(v):
    """Sigmoid activation function. Vectorised."""
    return 1.0 / (1.0 + np.exp(-v))
    # return np.tanh(v)


def _activation_prime(v):
    """Derivative of the sigmoid activation function. Vectorised."""
    return _activation(v) * (1 - _activation(v))
    # return 1 / np.power(np.tanh(v), 2)


def _cost_prime(Y_predicted, Y_actual):
    """Return the deriv. of the cost of each label w.r.t. its gold label.

    Original cost function is quadratic cost.

    Note that this returns a matrix of partial derivatives,
    """
    return Y_predicted - Y_actual


class FeedForwardNet(object):

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

    def train(self, X, Y):
        """Train given a set of data points, X, and gold labels, Y.

        Currently calculates a crude non-confidence weighted accuracy.
        """
        Y_probabilities = _to_categorical_identity(Y)

        for i in range(0, _TRAINING_EPOCHS):
            self._forward_pass(X)
            # TODO: Restore softmax output layer.
            # softmax_outputs = _softmax(self.A_o)
            # cost = _cost_prime(softmax_outputs, Y_probabilities)
            cost = _cost_prime(self.A_o, Y_probabilities)
            self._back_propagate(cost, X)

        # TODO: Split this into test and evaluate methods.
        print('Accuracy: {}'.format(
            np.mean(
                np.sum(
                    np.multiply(_take_max(self.A_o),
                                Y_probabilities), axis=0))))

    def _forward_pass(self, X):
        """Forward pass all points in X through the network.

        Stores both weighted inputs and activations for each layer.
        """
        self.Z_h = self.W_x_h.dot(X) + self.b_h
        self.A_h = _activation(self.Z_h)

        self.Z_o = self.W_h_o.dot(self.A_h) + self.b_o
        self.A_o = _activation(self.Z_o)

    def _back_propagate(self, cost, X):
        """Update weights and biases according to the cost.

        Uses backpropagation to apportion cost to each parameter.

        Currently also performs batch (that's right, the whole batch) gradient
        descent to actually update the parameters. We should probably just
        calculate the error of each layer here.
        """
        output_layer_error = np.multiply(
            cost, _activation_prime(self.Z_o))
        hidden_layer_error = np.multiply(self.W_h_o.T.dot(
            output_layer_error), _activation_prime(self.Z_h))

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
    n = FeedForwardNet(X_train.shape[0])
    n.train(X_train, Y_train)
