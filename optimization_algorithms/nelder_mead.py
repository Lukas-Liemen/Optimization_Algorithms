import numpy as np
from matplotlib import animation, pyplot as plt


class NelderMead:
    def __init__(self, fct: callable, dim: int, epsilon: float = 1e-7):
        """
        Perform the Nelder-Mead optimization algorithm.

        :param fct: The function to optimize.
        """

        self.fct = fct
        self.dim = dim
        self.epsilon = epsilon
        self.points = self.rdm_points()
        self.best_point_history = []
        self.simplex_history = []

    def rdm_points(self):
        """
        Generate random points to form simplex to start the optimization.

        :return: Array of random points.
        """

        points = np.random.rand(self.dim + 1, self.dim)
        return self.sort_points(points)

    def sort_points(self, points):
        """
        Sort the points based on the function value. Lower values first.

        :param points: Points to sort.
        :return: Sorted points.
        """
        eval = np.array([self.fct(*point) for point in points])
        points = points[np.argsort(eval)]
        return points

    def __call__(self):
        """
        Perform the Nelder-Mead optimization algorithm.

        :return: The optimum point.
        """

        while not self.convergence_crit():
            self.points = self.sort_points(self.points)
            reflection = self.next_point(-1)

            if self.fct(*self.points[0]) <= self.fct(*reflection) < self.fct(*self.points[-2]):
                self.points[-1] = reflection

            elif self.fct(*reflection) < self.fct(*self.points[0]):
                expansion = self.next_point(-2)
                if self.fct(*expansion) < self.fct(*reflection):
                    self.points[-1] = expansion
                else:
                    self.points[-1] = reflection

            elif self.fct(*self.points[-2]) <= self.fct(*reflection) < self.fct(*self.points[-1]):
                outside_contraction = self.next_point(-1/2)
                if self.fct(*outside_contraction) <= self.fct(*reflection):
                    self.points[-1] = outside_contraction
                else:
                    self.points = 0.5 * (self.points + self.points[0])

            else:
                inside_contraction = self.next_point(1/2)
                if self.fct(*inside_contraction) < self.fct(*self.points[-1]):
                    self.points[-1] = inside_contraction
                else:
                    self.points = 0.5 * (self.points + self.points[0])

            best_point = np.mean(self.points[:-1], axis=0)
            self.best_point_history.append(best_point)
            self.simplex_history.append(self.points)

        return best_point

    def convergence_crit(self):
        """
        Check the convergence criteria for the Nelder-Mead algorithm.

        :return: True if the convergence criteria is met.
        """

        centroid = self.centroid()
        centroid_val = self.fct(*centroid)

        sum = 0
        for i in range(len(self.points)):
            sum += (self.fct(*self.points[i]) - centroid_val) ** 2 / len(self.points)
        return np.sqrt(sum) < self.epsilon

    def centroid(self):
        """
        Calculate the centroid of the simplex.

        :return: The centroid of the simplex.
        """

        return np.mean(self.points[:-1], axis=0)

    def next_point(self, nature: float) -> np.ndarray:
        """
        Calculates reflection, expansion or contraction of the simplex.

        :param nature: Nature of the operation. -1 = reflection, -2 = expansion
                          -1/2 = outside contraction, 1/2 = inside contraction
        :return: The new point.
        """

        centroid = self.centroid()
        new_point = centroid + nature * (self.points[-1] - centroid)
        return new_point

    def show(self, ax):
        """
        Plot the Nelder-Mead optimization results on the given axis.

        :param ax: Axis to plot on.
        """

        optimum = self()
        self.best_point_history = np.array(self.best_point_history)
        ax.plot(self.best_point_history[:, 0], self.best_point_history[:, 1], "o-")
        ax.plot(optimum[0], optimum[1], "x", color="red")

        title = ax.get_title()
        ax.set_title(title + "\nNelder-Mead Optimization - Steps: {}".format(len(
            self.best_point_history)))

    def show_simplex(self, ax):
        """
        Plot the Nelder-Mead optimization results on the given axis.

        :param ax: Axis to plot on.
        """

        self()
        for simplex in self.simplex_history:
            ax.plot(simplex[:, 0], simplex[:, 1], "o-")
        ax.plot(self.best_point_history[-1][0], self.best_point_history[-1][1], "x", color="red")

    def animation(self, ax):
        """
        Creates an animation of the Nelder-Mead optimization results on the given axis.

        :param ax: Axis to plot on.
        :return: The animation object
        """

        # Run the optimization to populate simplex history
        self()

        # Initialize the polygon object for the animation with a dummy triangle
        # The initial triangle is a placeholder and will be updated in the first frame
        dummy_triangle = np.array([[0, 0], [0, 0], [0, 0]])
        simplex_patch = plt.Polygon(dummy_triangle, closed=True, color="red", alpha=0.5)

        # Add the polygon to the axis
        ax.add_patch(simplex_patch)

        # Function to initialize the plot
        def init():
            simplex_patch.set_xy(dummy_triangle)
            return simplex_patch,

        # Function to update the plot for each frame
        def update(frame):
            simplex = self.simplex_history[frame]
            simplex_patch.set_xy(simplex)
            return simplex_patch,

        # Create the animation
        anim = animation.FuncAnimation(
            fig=ax.figure,
            func=update,
            init_func=init,
            frames=len(self.simplex_history),
            interval=200,  # Adjust the interval as needed
            blit=True
        )

        return anim