"""A collection of derivatives of cost functions.

Note that we have no use for the undifferentiated cost function itself,
and as such they are omitted here.
"""


def quadratic_cost_prime(Y_predicted, Y_actual):
    """Return the deriv. of the cost of each label w.r.t. its gold label.

    Y_predicted is an m x n matrix of predicted labels,
    where m is the number of labels and n is the number of data points.

    Y_actual is an m x n matrix of actual labels.

    Returns an m x n matrix of partial derivatives.
    """
    return Y_predicted - Y_actual
