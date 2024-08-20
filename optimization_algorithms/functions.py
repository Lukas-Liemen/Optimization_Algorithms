import numpy as np


def line_search(fct, x, direction, num_points=1000):
    """
    Perform line search to minimize the function along a specified direction.
    Line search based on grid search.

    :param x: Current point.
    :param direction: Direction to search along.
    :param num_points: Number of points to search.
    :return: The new point after line search.
    """

    def fct_alpha(alpha):
        return fct(*(x + alpha * direction))

    alphas = np.linspace(-1, 1, num_points)
    values = [fct_alpha(alpha) for alpha in alphas]
    min_index = np.argmin(values)
    alpha = alphas[min_index]
    return x + alpha * direction


def calculate_length_of_vector(vector):
    """
    Calculate the length of a vector.

    :param vector: Vector to calculate the length of.
    :return: The length of the vector.
    """
    return np.sqrt(np.sum(np.square(vector)))**2
