import numpy as np

from optimization_algorithms.functions import line_search


class SteepestDescent:
    def __init__(self, fct: callable, grad_fct: callable, x0: np.ndarray, min_error: float = 1e-7):
        """
        Perform the steepest descent optimization algorithm.

        The steepest descent algorithm is a first-order optimization algorithm that uses the
        gradient of the function to find the optimum.

        :param fct: The function to optimize.
        :param grad_fct: The gradient of the function to optimize.
        :param x0: The initial guess for the optimization.
        :param min_error: minimum error for convergence.
        """
        self.fct = fct
        self.grad_fct = grad_fct
        self.x0 = x0
        self.min_error = min_error

        self.best_point_history = [self.x0.copy()]

    def __call__(self):
        self._main_loop()

        while np.linalg.norm(self.best_point_history[-1] -
                             self.best_point_history[-2]) > self.min_error:
            self._main_loop()

        return self.x0

    def _main_loop(self):
        dir = -self.grad_fct(*self.x0)
        self.x0 = line_search(self.fct, self.x0, dir)
        self.best_point_history.append(self.x0.copy())

    def show(self, ax):
        """
        Plot the steepest descent optimization results on the given axis.

        :param ax: Axis to plot on.
        """

        optimum = self()
        ax.plot([point[0] for point in self.best_point_history],
                [point[1] for point in self.best_point_history],
                "o-")
        ax.plot(optimum[0], optimum[1], "x", color="red")

        title = ax.get_title()
        ax.set_title(title + "\nSteepest Descent Optimization - Steps: {}".format(len(
            self.best_point_history)))
