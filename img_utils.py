import numpy as np
import matplotlib.pyplot as plt


def scatter_into_array(data, title):
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches((7, 7))
    ax.scatter(data[:, 0], data[:, 1], c="black", s=1)
    #fig.add_axes(ax)
    fig.canvas.draw()
    #plt.title(title)
    #plt.xlabel("First Dimension")
    #plt.ylabel("Second Dimension")
    return np.array(fig.canvas.renderer._renderer)
