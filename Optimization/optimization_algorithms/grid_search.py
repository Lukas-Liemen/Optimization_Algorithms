import numpy as np
from matplotlib import pyplot as plt


class GridSearch:
    def __init__(self, fct: callable, param_grid: dict):
        """
        Perform a grid search over the parameter grid for the given function.

        Grid search works by evaluating the function for each set of parameters in the grid.
        This makes it a simple but computationally expensive method for hyperparameter tuning.

        :param fct: The function to perform the grid search on.
        :param param_grid: The parameter grid to search over.
        :param verbose: Whether to print the search results.
        :return: The best parameters found during the search.
        """

        self.fct = fct
        self.param_grid = param_grid

    def __call__(self):
        best_params = {}
        best_score = np.inf

        for params in self.param_grid:
            score = self.fct(**params)
            if score < best_score:
                best_score = score
                best_params = params

        return best_params

    def show(self, ax: plt.Axes):
        """
        Plot the grid search results on the given axis.

        :param ax: Axis to plot on.
        """
        best_params = self()

        for params in self.param_grid:
            color = "green" if params != best_params else "red"
            ax.plot(params["x1"], params["x2"], "o", color=color)

        title = ax.get_title()
        ax.set_title(title + "\nGrid Search - Points Considered: {}".format(len(self.param_grid)))