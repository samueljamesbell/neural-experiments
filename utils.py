"""Assorted functions."""

import numpy as np


def softmax(M):
    """Compute Softmax of a given matrix, M."""
    M_exponents = np.exp(M)
    denominators = np.sum(M_exponents, axis=0)
    return np.divide(M_exponents, denominators)


def to_categorical_identity(Y, labels):
    """Return a sparse representation of a set of labels.

    For a given label y, return a vector with all probability mass assigned
    to the yth index of the vector.

    Labels should be the set of all possible gold labels.

    e.g. y = 2 could yield [0, 0, 1]
    """
    Y_categorical = np.zeros([Y.shape[0], len(labels)])
    for l in labels:
        Y_categorical[:, l] = Y == l

    return Y_categorical.transpose()


def take_max(M):
    """Given a matrix M, find the max values in each row.

    Returns a 1 in the column with the max value; 0 otherwise.
    """
    M_max = np.argmax(M, axis=0)
    N = np.zeros_like(M)
    for i in range(0, M.shape[0]):
        N[i, :] = M_max == i

    return N
