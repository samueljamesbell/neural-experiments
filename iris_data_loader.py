import numpy as np
import pandas


_TRAINING_PATH = 'data/iris_training.csv'
_TEST_PATH = 'data/iris_test.csv'

_FEATURE_NAMES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_COLUMN_NAMES = _FEATURE_NAMES + ['species']

LABELS = [0, 1, 2]


def _load_data(path):
    data = pandas.read_csv(path, names=_COLUMN_NAMES, skiprows=[0])
    # We transpose what's loaded from the CSV because we want each data point
    # to be a column vector, where each entry in the column vector is a
    # feature.
    X = data.loc[:, _FEATURE_NAMES].T
    Y = np.array([data.loc[:, 'species']])
    return X, Y


def training_data():
    """Return training data, as tuple (X, Y) of data points and labels.

    X is an f x n matrix, where f is the number of features, and n is the
    number of data points.

    Y is a 1 x n matrix.
    """
    return _load_data(_TRAINING_PATH)


def test_data():
    """Return test data, as tuple (X, Y) of data points and labels.

    X is an f x n matrix, where f is the number of features, and n is the
    number of data points.

    Y is a 1 x n matrix.
    """
    return _load_data(_TEST_PATH)
