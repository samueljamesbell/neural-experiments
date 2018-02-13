"""A collection of activation functions and their derivatives."""

import numpy as np


def sigmoid(M):
    """Sigmoid activation function.
    
    M is a d x n matrix where d is the dimensionality of the weighted input
    vectors, and n is the number of data points.

    Returns d x n matrix where sigmoid has been applied column-wise.
    """
    return 1.0 / (1.0 + np.exp(-M))


def sigmoid_prime(M):
    """Derivative of the sigmoid activation function.
    
    M is a d x n matrix where d is the dimensionality of the weighted input
    vectors, and n is the number of data points.

    Returns d x n matrix where sigmoid prime has been applied column-wise.
    """
    return sigmoid(M) * (1 - sigmoid(M))


def tanh(M):
    """Tanh activation function.
    
    M is a d x n matrix where d is the dimensionality of the weighted input
    vectors, and n is the number of data points.

    Returns d x n matrix where tanh has been applied column-wise.
    """
    return np.tanh(M)


def tanh_prime(M):
    """Derivative of the tanh activation function.
    
    M is a d x n matrix where d is the dimensionality of the weighted input
    vectors, and n is the number of data points.

    Returns d x n matrix where tanh prime has been applied column-wise.
    """
    return 1 / np.power(np.tanh(M), 2)


SIGMOID = sigmoid, sigmoid_prime
TANH = tanh, tanh_prime
