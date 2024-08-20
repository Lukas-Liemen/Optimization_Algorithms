import numpy as np
from matplotlib import pyplot as plt

from optimization_algorithms.functions import line_search


class Powell:
    def __init__(self, fct: callable, x0: np.ndarray, epsilon: float = 1e-7,
                 min_error: float = 1e-7):
        """
        Perform the Powell optimization algorithm.

        The Powell algorithm is a conjugate direction method that does not require the gradient
        of the function to be optimized.

        :param fct: The function to optimize.
        :param x0: The initial guess for the optimization.
        :param epsilon: probing step size.
        :param min_error: minimum error for convergence.
        """
        self.fct = fct
        self.x0 = x0
        self.epsilon = epsilon
        self.min_error = min_error

        self.best_point_history = [self.x0.copy()]

    def __call__(self):
        directions = np.eye(self.x0.size)

        # Univariate Search - Cycle through each direction
        for dir in directions:
            dir = self._check_direction(dir)
            new_point = line_search(self.fct, self.x0, dir)
            self.best_point_history.append(new_point.copy())
            self.x0 = new_point

        # Pattern Search
        while np.linalg.norm(self.best_point_history[-1] -
                             self.best_point_history[-2]) > self.min_error:
            dir = self.best_point_history[-1] - self.best_point_history[-3]
            dir = self._check_direction(dir)
            new_point = line_search(self.fct, self.x0, dir)
            self.best_point_history.append(new_point.copy())
            self.x0 = new_point

            directions = np.vstack((directions[1:], dir))

            for dir in directions:
                new_point = line_search(self.fct, self.x0, dir)
                self.best_point_history.append(new_point.copy())
                self.x0 = new_point

        return self.x0

    def _check_direction(self, direction: np.ndarray):
        """
        Check if the direction is a descent direction.

        :param direction: Direction to check.
        :return: True if the direction is a descent direction, False otherwise.
        """

        new_point = self.x0 + self.epsilon * direction
        if self.fct(*new_point) < self.fct(*self.x0):
            return direction
        else:
            return -direction

    def show(self, ax: plt.Axes):
        """
        Plot the Powell optimization results on the given axis.

        :param ax: Axis to plot on.
        """
        best_point = self()
        self.best_point_history = np.array(self.best_point_history)
        ax.plot(self.best_point_history[:, 0], self.best_point_history[:, 1], "o-")
        ax.plot(best_point[0], best_point[1], "x", color="red")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        title = ax.get_title()
        ax.set_title(title + "\nPowell Optimization - Steps: {}".format(len(self.best_point_history)))
