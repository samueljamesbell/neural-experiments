import numpy as np
import pandas
import math


_INPUT_LAYER_DIMENSIONS = 4
_HIDDEN_LAYER_DIMENSIONS = 4 
_OUTPUT_LAYER_DIMENSIONS = 3

_TRAINING_EPOCHS = 1000
_LEARNING_RATE = 0.0001

_TRAINING_PATH = 'data/iris_training.csv'
_TEST_PATH = 'data/iris_test.csv'

_FEATURE_NAMES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_COLUMN_NAMES = _FEATURE_NAMES + ['species']
_SPECIES = ['sentosa', 'versicolor', 'virginica']


def _load_data(path):
    data = pandas.read_csv(path, names=_COLUMN_NAMES, skiprows=[0])
    X = data.loc[:, _FEATURE_NAMES].T
    Y = data.loc[:, 'species']
    return X, Y 


class FeedForwardNet(object):

    def __init__(self, number_training_examples):
        self.W_x_h = np.random.rand(_HIDDEN_LAYER_DIMENSIONS,
                _INPUT_LAYER_DIMENSIONS)
        self.W_h_o = np.random.rand(_OUTPUT_LAYER_DIMENSIONS,
                _HIDDEN_LAYER_DIMENSIONS)
        self.b_h = np.random.rand(_HIDDEN_LAYER_DIMENSIONS, 1)
        self.b_o = np.random.rand(_OUTPUT_LAYER_DIMENSIONS, 1)

        self.Z_h = np.zeros([number_training_examples, _HIDDEN_LAYER_DIMENSIONS])
        self.Z_o = np.zeros([number_training_examples, _OUTPUT_LAYER_DIMENSIONS])

        self.A_h = np.zeros([number_training_examples, _HIDDEN_LAYER_DIMENSIONS])
        self.A_o = np.zeros([number_training_examples, _OUTPUT_LAYER_DIMENSIONS])

    def train(self, X, Y):
        Y_probabilities = self._gold_label_probabilities(Y)

        for i in range(0, _TRAINING_EPOCHS):
            self._forward_pass(X)
            softmax_outputs = self._softmax(self.A_o)
            cost = self._cost_prime(softmax_outputs, Y_probabilities)
            self._back_propagate(cost, X)

        print('Accuracy: {}'.format(np.mean(np.multiply(softmax_outputs,
            Y_probabilities))))


    def _gold_label_probabilities(self, Y):
        """Return a vector representation of a label.

        For a given label y, return a vector with all probability density assigned
        to the yth index of the vector.

        e.g.
        y = 2
        =>
        v = [0, 0, 1]
        """
        Y_probabilities = np.zeros([Y.shape[0], len(np.unique(Y))])
        Y_probabilities[:, 0] = Y == 0
        Y_probabilities[:, 1] = Y == 1
        Y_probabilities[:, 2] = Y == 2
        return Y_probabilities.transpose()

    def _forward_pass(self, X):
        self.Z_h = self.W_x_h.dot(X) + self.b_h
        self.A_h = self._activation(self.Z_h)

        self.Z_o = self.W_h_o.dot(self.A_h) + self.b_o
        self.A_o = self._activation(self.Z_o)

    def _back_propagate(self, cost, X):
        output_layer_error = np.multiply(cost, self._activation_prime(self.Z_o))
        hidden_layer_error = np.multiply(self.W_h_o.T.dot(output_layer_error), self._activation_prime(self.Z_h))

        output_layer_bias_gradients = output_layer_error
        hidden_layer_bias_gradients = hidden_layer_error

        output_layer_weight_gradients = output_layer_error.dot(self.A_h.T)
        hidden_layer_weight_gradients = hidden_layer_error.dot(X.T)

        ### This is batch gradient descent - should probably be its own method
        self.b_o = self.b_o - (np.mean(output_layer_bias_gradients,
                                      axis=1).reshape(self.b_o.shape[0], 1) *
                                      _LEARNING_RATE)
        self.b_h = self.b_h - (np.mean(hidden_layer_bias_gradients,
                                      axis=1).reshape(self.b_h.shape[0], 1) *
                                      _LEARNING_RATE)

        self.W_h_o = self.W_h_o - (np.mean(output_layer_weight_gradients,
            axis=0) * _LEARNING_RATE)
        self.W_x_h = self.W_x_h - (np.mean(hidden_layer_weight_gradients,
            axis=0) * _LEARNING_RATE)

    def _softmax(self, M):
        """Compute Softmax of a given matrix, M."""
        M_exponents = np.exp(M)
        denominators = np.sum(M_exponents, axis=0)
        return np.divide(M_exponents, denominators)

    def _activation(self, v):
        return np.tanh(v)

    def _activation_prime(self, v):
        return 1 / np.power(np.tanh(v), 2)

    def _cost_prime(self, Y_predicted, Y_actual):
        """Return the deriv. of the cost of each label w.r.t. its gold label.
        
        Note that this returns a matrix of partial derivatives,
        """
        # TODO: Specify the cost function in the comments
        #indices = np.argmax(Y_predicted, axis=0)
        return Y_predicted - Y_actual


X_train, Y_train = _load_data(_TRAINING_PATH)
n = FeedForwardNet(X_train.shape[0])
n.train(X_train, Y_train)
