"""A collection of activation functions and their derivatives."""

import numpy as np


def sigmoid(v):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-v))


def sigmoid_prime(v):
    """Derivative of the sigmoid activation function."""
    return sigmoid(v) * (1 - sigmoid(v))


def tanh(v):
    """Tanh activation function."""
    return np.tanh(v)


def tanh_prime(v):
    """Derivative of the tanh activation function."""
    return 1 / np.power(np.tanh(v), 2)


SIGMOID = sigmoid, sigmoid_prime
TANH = tanh, tanh_prime
