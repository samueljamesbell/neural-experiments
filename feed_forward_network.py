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

    def __init__(self, number_training_examples):
        self.W_x_h = np.random.randn(_HIDDEN_LAYER_DIMENSIONS,
                                     _INPUT_LAYER_DIMENSIONS)
        self.W_h_o = np.random.randn(_OUTPUT_LAYER_DIMENSIONS,
                                     _HIDDEN_LAYER_DIMENSIONS)
        self.b_h = np.random.randn(_HIDDEN_LAYER_DIMENSIONS, 1)
        self.b_o = np.random.randn(_OUTPUT_LAYER_DIMENSIONS, 1)

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
            deltas = self._back_propagate(C, X, Z, A)
            self._batch_gradient_descent(deltas, X, A, learning_rate)

        # Perform a final forward pass with our optimised weights.
        _, A = self._forward_pass(X)
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
        A = []

        Z.append(self.W_x_h.dot(X) + self.b_h)
        A.append(activation.sigmoid(Z[0]))

        Z.append(self.W_h_o.dot(A[0]) + self.b_o)
        A.append(activation.sigmoid(Z[1]))

        return Z, A

    def _back_propagate(self, C, X, Z, A):
        """Calculate error w.r.t C for each layer.

        C is an o x n matrix, where o is the dimensionality of the output
        layer and n is the number of training examples. C represents a matrix
        of partial derivatives, where every column is the partial derivative
        of the cost function with respect to each entry in the output
        activation vector.

        X is an f x n matrix, where f is the number of features and n is the
        number of training examples.

        Z is a list of weighted inputs, where each
        entry is a d x n matrix, where d is the dimensionality of the layer and
        n is the number of data points.

        A is a list of activations, with the same shape as Z.

        The final item in A is the output layer, and the first item is the
        first hidden layer. The same is true for Z.

        Returns deltas, a list of errors where each entry in the list is a
        1 x d matrix where d is the dimensionality of the layer, representing
        the error w.r.t each activation in the layer.
        """
        deltas = []

        deltas.append(np.multiply(C, activation.sigmoid_prime(Z[-1])))
        deltas.append(np.multiply(self.W_h_o.T.dot(
            deltas[0]), activation.sigmoid_prime(Z[-2])))

        # We reverse the order of deltas so its indices align with A.
        # That is, so that the error delta at index i is the error of the A[i].
        return deltas[::-1]

    def _batch_gradient_descent(self, deltas, X, A, learning_rate):
        """Update weights according to the layer-by-layer errors.

        deltas is a list of errors where each entry in the list is a
        1 x d matrix where d is the dimensionality of the layer, representing
        the error w.r.t each activation in the layer.

        X is an f x n matrix, where f is the number of features and n is the
        number of training examples.

        A is a list of activations, where each entry is a d x n matrix, where
        d is the dimensionality of the layer and n is the number of data
        points.

        Currently performs batch (that's right, the whole batch) gradient
        descent to actually update the parameters.
        """
        output_layer_bias_gradients = deltas[-1]
        hidden_layer_bias_gradients = deltas[-2]

        output_layer_weight_gradients = deltas[-1].dot(A[-2].T)
        hidden_layer_weight_gradients = deltas[-2].dot(X.T)

        self.b_o = self.b_o - (np.mean(output_layer_bias_gradients,
                                       axis=1).reshape(self.b_o.shape[0], 1) *
                               learning_rate)
        self.b_h = self.b_h - (np.mean(hidden_layer_bias_gradients,
                                       axis=1).reshape(self.b_h.shape[0], 1) *
                               learning_rate)

        self.W_h_o = self.W_h_o - \
            (output_layer_weight_gradients * learning_rate)
        self.W_x_h = self.W_x_h - \
            (hidden_layer_weight_gradients * learning_rate)
