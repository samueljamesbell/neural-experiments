import pandas


_TRAINING_PATH = 'data/iris_training.csv'
_TEST_PATH = 'data/iris_test.csv'

_FEATURE_NAMES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_COLUMN_NAMES = _FEATURE_NAMES + ['species']

LABELS = [0, 1, 2]


def _load_data(path):
    data = pandas.read_csv(path, names=_COLUMN_NAMES, skiprows=[0])
    X = data.loc[:, _FEATURE_NAMES].T
    Y = data.loc[:, 'species']
    return X, Y


def training_data():
    """Return training data, as tuple of data points and labels."""
    return _load_data(_TRAINING_PATH)


def test_data():
    """Return test data, as tuple of data points and labels."""
    return _load_data(_TEST_PATH)
