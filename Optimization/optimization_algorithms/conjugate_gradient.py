import numpy as np

from optimization_algorithms.functions import line_search, calculate_length_of_vector


class ConjugateGradient:
    def __init__(self, fct: callable, grad_fct: callable, x0: np.ndarray, min_error: float = 1e-7):
        """
        Perform the conjugate gradient optimization algorithm.

        The conjugate gradient algorithm is a first-order optimization algorithm that uses the
        gradient of the function to find the optimum. This specific implementation uses the
        Fletcher-Reeves method.

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
        search_direction = -self.grad_fct(*self.x0)
        search_direction, error = self._main_loop(search_direction)

        while error > self.min_error:
            search_direction, error = self._main_loop(search_direction)

        return self.x0

    def _main_loop(self, search_direction):
        dist_cur = calculate_length_of_vector(self.grad_fct(*self.x0))
        self.x0 = line_search(self.fct, self.x0, search_direction)
        self.best_point_history.append(self.x0.copy())

        grad = self.grad_fct(*self.x0)
        dist_new = calculate_length_of_vector(grad)
        search_direction = -grad + (dist_new / dist_cur) * search_direction  # Fletcher-Reeves
        error = np.linalg.norm(self.best_point_history[-1] - self.best_point_history[-2])

        return search_direction, error

    def show(self, ax):
        """
        Plot the conjugate gradient optimization results on the given axis.

        :param ax: Axis to plot on.
        """

        optimum = self()
        ax.plot([point[0] for point in self.best_point_history],
                [point[1] for point in self.best_point_history],
                "o-")
        ax.plot(optimum[0], optimum[1], "x", color="red")

        title = ax.get_title()
        ax.set_title(title + "\nConjugate Gradient Optimization - Steps: {}".format(len(
            self.best_point_history)))
