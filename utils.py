"""Assorted functions."""

import numpy as np


def softmax(M):
    """Compute Softmax of a given matrix, M.

    M is an m x n matrix where m is typically the number of output activations,
    and n is the number of data points. Thus we have n column vectors, each of
    m dimensions.

    Returns an m x n matrix where softmax is applied column-wise.
    """
    M_exponents = np.exp(M)
    denominators = np.sum(M_exponents, axis=0)
    return np.divide(M_exponents, denominators)


def accuracy(Y_predicted, Y_actual):
    """Return classification acc. from two categorical identity matrices.

    Y_predicted is an m x n categorical identity matrix of predicted labels,
    where m is the number of labels and n is the number of data points.

    Y_actual is an m x n categorical identity matrix of actual labels.

    Returns a real number in the range [0, 1].
    """
    return np.mean(np.sum(np.multiply(Y_predicted, Y_actual), axis=0))


def to_categorical_identity(Y, labels):
    """Return a sparse representation of a set of labels.

    Y should be a 1 x n matrix, where n is the number of data points.

    Returns an m x n matrix, where m is the number of discrete labels.

    For a given label y, we construct a column vector with all probability mass
    assigned to the yth index of the vector.

    Labels should be the set of all possible gold labels.

    e.g. `to_categorical_identity([[2]], [0, 1, 2]) => [[0], [0], [1]]`
    """
    Y_categorical = np.zeros([len(labels), Y.shape[1]])
    for l in labels:
        Y_categorical[l, :] = Y == l

    return Y_categorical


def take_max(M):
    """Given an m x n matrix M, find the max values in each column.

    Returns an m x n matrix, where the position of the maximum value each
    column of M is marked by a 1; 0 otherwise.
    """
    M_max = np.argmax(M, axis=0)
    N = np.zeros_like(M)
    for i in range(0, M.shape[0]):
        N[i, :] = M_max == i

    return N
