import matplotlib.pyplot as plt
import numpy as np
import settings
from force import pbc
from numba import njit, prange


@njit
def histogram(x, y, z, bin_width, rmax):
    # calc the distance between the particles
    # only consider distances less than rmax
    hist = np.zeros(int(rmax / bin_width))
    for i in prange(len(x)):
        for j in prange(i + 1, len(x)):
            # rijx = x[i] - x[j]
            # rijy = y[i] - y[j]
            # rijz = z[i] - z[j]
            rijx = pbc(x[i], x[j], settings.xlo, settings.xhi)
            rijy = pbc(y[i], y[j], settings.ylo, settings.yhi)
            rijz = pbc(z[i], z[j], settings.zlo, settings.zhi)

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


def plot_rdf(rdf, bin_width):
    x = np.arange(0, len(rdf)) * bin_width / settings.sigma
    box_length_sigma = settings.xhi / settings.sigma / 2
    other_line = np.sqrt(2) * box_length_sigma
    plt.axvline(other_line, color="g", linestyle="--", label="sqrt(2) * box length")
    plt.axvline(box_length_sigma, color="r", linestyle="--", label="box length")
    plt.plot(x, rdf, label="g(r)")
    plt.xlabel("r/ sigma")
    plt.ylabel("g(r)")
    plt.legend()
    plt.show()
