import matplotlib.pyplot as plt
import numpy as np

from optimization_algorithms.bfgs import BFGS
from optimization_algorithms.conjugate_gradient import ConjugateGradient
from optimization_algorithms.grid_search import GridSearch
from optimization_algorithms.nelder_mead import NelderMead
from optimization_algorithms.powell import Powell
from optimization_algorithms.steepest_descent import SteepestDescent


def himmelblau(x1, x2):
    """
    Himmelblau function.
    """
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


def grad_himmelblau(x1, x2):
    """
    Gradient of the Himmelblau function.
    """
    grad_x1 = 2 * (x1**2 + x2 - 11) * 2*x1 + 2 * (x1 + x2**2 - 7)
    grad_x2 = 2 * (x1**2 + x2 - 11) + 2 * (x1 + x2**2 - 7) * 2*x2
    return np.array([grad_x1, grad_x2])


def plot_himmelblau(ax, min=-5, max=5, num=100):
    """
    Plot the Himmelblau function.

    :param ax: Axis to plot on.
    :param min: Minimum value for the x and y axes.
    :param max: Maximum value for the x and y axes.
    :param num: Number of points to plot.
    """

    x1 = np.linspace(min, max, num)
    x2 = np.linspace(min, max, num)
    x1, x2 = np.meshgrid(x1, x2)
    z = himmelblau(x1, x2)
    ax.contourf(x1, x2, z, levels=50, cmap="viridis")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Himmelblau Function")


if __name__ == "__main__":
    save_folder = "./images"
    x0 = np.array([0, 0])
    param_grid = [{"x1": x1, "x2": x2} for x1 in np.linspace(-4, 4, 10)
                  for x2 in np.linspace(-4, 4, 10)]

    # Perform grid search
    fig, ax = plt.subplots()
    plot_himmelblau(ax)

    # Define a list of tuples containing the optimizer name, optimizer instance, and save filename
    optimizers = [
        ("grid_search", GridSearch(himmelblau, param_grid)),
        ("powell", Powell(himmelblau, x0)),
        ("steepest_descent", SteepestDescent(himmelblau, grad_himmelblau, x0)),
        ("conjugate_gradient", ConjugateGradient(himmelblau, grad_himmelblau, x0)),
        ("nelder_mead", NelderMead(himmelblau, dim=2)),
        ("bfgs", BFGS(himmelblau, grad_himmelblau, x0)),
    ]

    for name, optimizer in optimizers:
        fig, ax = plt.subplots()
        plot_himmelblau(ax)
        optimizer.show(ax)
        plt.savefig(f"{save_folder}/{name}.png")
        plt.close(fig)  # Close the figure after saving to free up memory

    # generate nelder mead animation
    fig, ax = plt.subplots()
    plot_himmelblau(ax)
    nelder_mead = NelderMead(himmelblau, dim=2)
    anim = nelder_mead.animation(ax)
    anim.save(f"{save_folder}/nelder_mead.gif", writer="imagemagick")