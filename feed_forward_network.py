"""A simple feedforward neural network with a fixed number of layers. """
import numpy as np

import activation
import cost
import utils


_INPUT_LAYER_DIMENSIONS = 4
_HIDDEN_LAYER_DIMENSIONS = 3
_OUTPUT_LAYER_DIMENSIONS = 3


class FeedForwardNet(object):

    # TODO: Generalise to flexible number of layers and dimensions.
    # TODO: Use softmax output layer and cross-entropy loss function.

    def __init__(self, layers):
        """Initialise the network.

        layers should be a list of layer sizes. We assume the first entry is
        the size the input vector, and the last entry is the size of the
        output vector.
        """
        self.W = self._weights(layers)
        self.b = self._biases(layers)
        self.num_layers = len(layers)

    def _weights(self, layers):
        """Initialise a random set of weight matrices."""
        return [np.random.randn(layers[i+1], layers[i])
                for i in range(0, len(layers) - 1)]

    def _biases(self, layers):
        """Initialise a random set of bias vectors."""
        # There are no biases for the input vector
        return [np.random.randn(l, 1) for l in layers[1:]]

    def train(self, X, Y, training_epochs=1, learning_rate=0.0001):
        """Train given a set of data points, X, and gold labels, Y.

        X is an f x n matrix, where f is the number of features and n is the
        number of training examples.

        Y is an l x n matrix, where l is the number of gold labels.

        Returns an l x n matrix of predicted categorical labels.
        """
        for i in range(0, training_epochs):
            Z, A = self._forward_pass(X)
            # softmax_outputs = utils.softmax(self.A_o)
            # cost = cost.quadratic_cost_prime(softmax_outputs, Y)
            C = cost.quadratic_cost_prime(A[-1], Y)
            deltas = self._back_propagate(C, Z)
            self._batch_gradient_descent(deltas, A, learning_rate)

        # Perform a final forward pass with our optimised weights.
        return utils.take_max(A[-1])

    def test(self, X):
        """Test infererence given  a set of data points, X, and gold labels, Y.

        X is an f x n matrix, where f is the number of features and n is the
        number of training examples.

        Returns an l x n matrix of predicted categorical labels, where l is the
        number of gold labels.
        """
        _, A = self._forward_pass(X)
        return utils.take_max(A[-1])

    def _forward_pass(self, X):
        """Forward pass all points in X through the network.

        X is an f x n matrix, where f is the number of features and n is the
        number of training examples.

        Returns a tuple of Z, A. Z is a list of weighted inputs, where each
        entry is a d x n matrix, where d is the dimensionality of the layer and
        n is the number of data points. A is a list of activations, with the
        same shape as Z.
        """
        Z = []
        A = [X]

        for i in range(0, self.num_layers - 1):
            z = self.W[i].dot(A[i]) + self.b[i]
            a = activation.sigmoid(z)
            Z.append(z)
            A.append(a)

        return Z, A

    def _back_propagate(self, C, Z):
        """Calculate error w.r.t C for each layer.

        C is an o x n matrix, where o is the dimensionality of the output
        layer and n is the number of training examples. C represents a matrix
        of partial derivatives, where every column is the partial derivative
        of the cost function with respect to each entry in the output
        activation vector.

        Z is a list of weighted inputs, where each
        entry is a d x n matrix, where d is the dimensionality of the layer and
        n is the number of data points.

        The final item in Z is the output layer, and the first item is the
        first hidden layer.

        Returns deltas, a list of errors where each entry in the list is a
        1 x d matrix where d is the dimensionality of the layer, representing
        the error w.r.t each activation in the layer.
        """
        deltas = []

        # Calculate error with respect to the output layer
        deltas.append(np.multiply(C, activation.sigmoid_prime(Z[-1])))

        # Calculate error with respect to each hidden layer
        for i in range(0, self.num_layers - 2):
            i_reverse = self.num_layers - 2 - i
            deltas.append(np.multiply(self.W[i_reverse].T.dot(
                deltas[i]),
                activation.sigmoid_prime(Z[i_reverse-1])))

        return deltas

    def _batch_gradient_descent(self, deltas, A, learning_rate):
        """Update weights according to the layer-by-layer errors.

        deltas is a list of errors where each entry in the list is a
        1 x d matrix where d is the dimensionality of the layer, representing
        the error w.r.t each activation in the layer.

        A is a list of activations, where each entry is a d x n matrix, where
        d is the dimensionality of the layer and n is the number of data
        points.

        Currently performs batch (that's right, the whole batch) gradient
        descent to actually update the parameters.
        """
        for i in range(0, self.num_layers - 1):
            i_reverse = self.num_layers - 2 - i

            self.b[i_reverse] = self.b[i_reverse] - (np.mean(
                deltas[i_reverse], axis=1).reshape(
                    self.b[i_reverse].shape[0], 1) * learning_rate)

            self.W[i_reverse] = self.W[i_reverse] - deltas[i_reverse].dot(
                    A[i_reverse].T) * learning_rate
