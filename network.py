import numpy as np
import pandas


_INPUT_LAYER_DIMENSIONS = 4
_HIDDEN_LAYER_DIMENSIONS = 4 
_OUTPUT_LAYER_DIMENSIONS = 3

_TRAINING_PATH = 'data/iris_training.csv'
_TEST_PATH = 'data/iris_test.csv'

_FEATURE_NAMES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_COLUMN_NAMES = _FEATURE_NAMES + ['species']
_SPECIES = ['sentosa', 'versicolor', 'virginica']


def _load_data(path):
    data = pandas.read_csv(path, names=_COLUMN_NAMES, skiprows=[0])
    X = data.loc[:, _FEATURE_NAMES]
    Y = data.loc[:, 'species']
    return X, Y 


def _label_to_probability_density_function(y):
    """Return a vector representation of a label.
    
    For a given label y, return a vector with all probability density assigned
    to the yth index of the vector.

    e.g.
    y = 2
    =>
    v = [0, 0, 1]
    """
    v = np.zeros(3)
    v[y] = 1
    return v




class FeedForwardNet(object):

    def __init__(self, number_training_examples):
        self.W_x_h = np.random.rand(_HIDDEN_LAYER_DIMENSIONS, _INPUT_LAYER_DIMENSIONS)
        self.W_h_o = np.random.rand(_HIDDEN_LAYER_DIMENSIONS, _OUTPUT_LAYER_DIMENSIONS)
        self.b_h = np.random.rand(_HIDDEN_LAYER_DIMENSIONS)
        self.b_o = np.random.rand(_OUTPUT_LAYER_DIMENSIONS)

        self.Z_h = np.zeros([number_training_examples, _HIDDEN_LAYER_DIMENSIONS])
        self.Z_o = np.zeros([number_training_examples, _OUTPUT_LAYER_DIMENSIONS])

        self.A_h = np.zeros([number_training_examples, _HIDDEN_LAYER_DIMENSIONS])
        self.A_o = np.zeros([number_training_examples, _OUTPUT_LAYER_DIMENSIONS])

    def train(self, X, Y):
        self._forward_pass(X)
        softmax_outputs = self._softmax(self.A_o)
        Y_probabilities = self._gold_label_probabilities(Y)

    def _gold_label_probabilities(self, Y):
        Y_probabilities = np.zeros([Y.shape[0], len(np.unique(Y))])
        Y_probabilities[:, 0] = Y == 0
        Y_probabilities[:, 1] = Y == 1
        Y_probabilities[:, 2] = Y == 2
        return Y_probabilities

    def _forward_pass(self, X):
        self.Z_h = X.dot(self.W_x_h) + self.b_h
        self.A_h = np.squeeze(self._activation(self.Z_h))

        self.Z_o = self.A_h.dot(self.W_h_o) + self.b_o
        self.A_o = np.squeeze(self._activation(self.Z_o))

    def _softmax(self, M):
        """Compute Softmax of a given matrix, M."""
        M_length = M.shape[0]
        M_exponents = np.exp(M)
        denominators = np.sum(M_exponents, axis=1).values.reshape(M_length, 1)
        return np.divide(M_exponents, denominators)

    def _activation(self, v):
        return np.tanh(v)

    def _activation_prime(self, v):
        return 1 / np.pow(np.tanh(v), 2)

    def _quadratic_cost(y_predicted, y_actual):
        """Return the deriv. of the cost of a single label w.r.t. its gold label.
        
        Note that this returns a vector of partial derivatives,
        """
        # TODO: Specify the cost function
        return y_predicted - y_actual



X_train, Y_train = _load_data(_TRAINING_PATH)
n = FeedForwardNet(X_train.shape[0])
n.train(X_train, Y_train)
