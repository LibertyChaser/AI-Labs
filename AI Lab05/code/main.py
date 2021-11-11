import random

import matplotlib.pyplot as plt

from GeneticAlgorithm import *
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    a = GeneticAlgorithm()
    GeneticAlgorithm.GA(a)

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    x_1 = np.arange(a.LOWER, a.UPPER, 0.1)
    x_2 = np.arange(a.LOWER, a.UPPER, 0.1)
    x_1, x_2 = np.meshgrid(x_1, x_2)
    ax.plot_surface(x_1, x_2, fitness_function(x_1, x_2), rstride=1, cstride=1)
    ax.view_init(elev=30, azim=125)
    plt.show()
