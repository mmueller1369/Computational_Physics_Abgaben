import matplotlib.pyplot as plt
import numpy as np
import settings
from numba import njit, prange


@njit
def histogram(x, y, z, bin_width, rmax):
    # calc the distance between the particles
    # only consider distances less than rmax
    hist = np.zeros(int(rmax / bin_width))
    for i in prange(len(x)):
        for j in prange(i + 1, len(x)):
            rijx = x[i] - x[j]
            rijy = y[i] - y[j]
            rijz = z[i] - z[j]

            r2 = rijx * rijx + rijy * rijy + rijz * rijz
            if r2 < rmax * rmax:
                r = np.sqrt(r2)
                bin_n = int(r / bin_width)
                hist[bin_n] += 2
    # after sort the distance in the bins

    return hist


@njit
def calc_RDF(histogram, bin_width):
    # calculate the average number of atoms in each bin
    total_bins = histogram.shape[1]

    histogram_new = np.zeros(total_bins)
    for i in prange(total_bins):
        total_atoms = 0
        for j in prange(histogram.shape[0]):
            total_atoms += histogram[j][i]
        histogram_new[i] = total_atoms / settings.n_gr / settings.nsteps_production

    # calculate the n(b) idela gas
    histogram_ideal = np.zeros(total_bins)
    for i in prange(total_bins):
        histogram_ideal[i] = (
            4
            / 3
            * np.pi
            * settings.rho
            * ((i * bin_width + bin_width) ** 3 - (i * bin_width) ** 3)
        )
    return histogram_new / histogram_ideal, [histogram_new, histogram_ideal]


def plot_histogram(hist):
    # plot the histogram
    plt.plot(hist)
    plt.xlabel("r (nm)")
    plt.ylabel("g(r)")
    plt.title("test")
    plt.show()

    total_sum = np.sum(hist)
    print("Sum of all bins:", total_sum)


def plot_rdf(rdf):
    plt.plot(rdf)
    plt.show()
