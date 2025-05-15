import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange


def histogram(x, y, z, bin_width, rmax, hist):
    # calc the distance between the particles
    # only consider distances less than rmax
    for i in prange(len(x)):
        for j in prange(i + 1, len(x)):
            rijx = x[i] - x[j]
            rijy = y[i] - y[j]
            rijz = z[i] - z[j]

            r2 = rijx * rijx + rijy * rijy + rijz * rijz
            if r2 < rmax * rmax:
                r = np.sqrt(r2)
                bin_n = int(r / bin_width)
                hist[bin_n] += 1
    # after sort the distance in the bins

    return hist


def plot_histogram(hist, deltar, rmax):
    # plot the histogram

    y = hist
    plt.plot(hist)
    plt.xlabel("r (nm)")
    plt.ylabel("g(r)")
    plt.title("Radial Distribution Function")
    plt.show()

    total_sum = np.sum(hist)
    print("Sum of all bins:", total_sum)
