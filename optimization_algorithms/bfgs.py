import numpy as np

from optimization_algorithms.functions import line_search


class BFGS:
    def __init__(self, fct: callable, grad_fct: callable, x0: np.ndarray, epsilon: float = 1e-7):
        """
        Perform the Broyden-Fletcher-Goldfarb-Shanno optimization algorithm.

        The BFGS algorithm is a quasi-Newton method that approximates the Hessian matrix
        of the function to be optimized.

        :param fct: The function to optimize.
        :param grad_fct: The gradient of the function to optimize.
        :param dim: Dimension of the function to optimize.
        """

        self.fct = fct
        self.grad_fct = grad_fct
        self.x0 = x0
        self.dim = x0.size
        self.epsilon = epsilon

        self.B = np.eye(self.dim)
        self.best_point_history = [self.x0.copy()]

    def __call__(self):
        self._main_loop()

        while np.linalg.norm(self.best_point_history[-1] - self.best_point_history[-2]) > self.epsilon:
            self._main_loop()

        return self.x0

    def _main_loop(self):
        grad = self.grad_fct(*self.x0)
        dir = -np.dot(self.B, grad)
        new_x = line_search(self.fct, self.x0, dir)

        dk = new_x - self.x0
        gk = self.grad_fct(*new_x) - grad
        divisor = np.dot(dk, gk)

        first = (np.outer(dk, dk) / divisor) * (1 + np.dot(gk, np.dot(self.B, gk)) / divisor)
        second = np.outer(np.dot(self.B, gk), dk) / divisor
        third = np.outer(dk, np.dot(gk, self.B)) / divisor
        self.B += first - second - third

        self.x0 = new_x
        self.best_point_history.append(self.x0.copy())

    def show(self, ax):
        """
        Plot the BFGS optimization results on the given axis.

        :param ax: Axis to plot on.
        """

        optimum = self()
        ax.plot([point[0] for point in self.best_point_history],
                [point[1] for point in self.best_point_history],
                "o-")
        ax.plot(optimum[0], optimum[1], "x", color="red")
        title = ax.get_title()
        ax.set_title(title + "\nBFGS Optimization - Steps: {}".format(len(self.best_point_history)))