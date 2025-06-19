import matplotlib.pyplot as plt
import numpy as np
import settings
import os
from forces import pbc
from numba import njit, prange
import os


@njit
def histogram(x, y, z, bin_width, rmax):
    # calc the distance between the particles
    # only consider distances less than rmax
    hist = np.zeros(int(rmax / settings.deltar))
    for i in prange(len(x)):
        for j in prange(i + 1, len(x)):
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


def histogramBond(x, y, z, bin_width, rmax):
    # Berechne nur die Abstände zwischen den Schwerpunkten (COMs) der Moleküle
    # Annahme: Atom 0+1 = Molekül 0, 2+3 = Molekül 1, ...
    n_mol = len(x) // 2
    com_x = np.zeros(n_mol)
    com_y = np.zeros(n_mol)
    com_z = np.zeros(n_mol)
    for i in range(n_mol):
        com_x[i] = 0.5 * (x[2 * i] + x[2 * i + 1])
        com_y[i] = 0.5 * (y[2 * i] + y[2 * i + 1])
        com_z[i] = 0.5 * (z[2 * i] + z[2 * i + 1])
    hist = np.zeros(int(rmax / settings.deltar))
    for i in prange(n_mol):
        for j in prange(i + 1, n_mol):
            rijx = pbc(com_x[i], com_x[j], settings.xlo, settings.xhi)
            rijy = pbc(com_y[i], com_y[j], settings.ylo, settings.yhi)
            rijz = pbc(com_z[i], com_z[j], settings.zlo, settings.zhi)
            r2 = rijx * rijx + rijy * rijy + rijz * rijz
            if r2 < rmax * rmax:
                r = np.sqrt(r2)
                bin_n = int(r / bin_width)
                hist[bin_n] += 2
    return hist


@njit
def calc_RDF(histogram, bin_width):
    # calculate the average number of atoms in each bin
    total_bins = histogram.shape[1]
    nparticles = settings.n1 * settings.n2 * settings.n3
    histogram_new = np.zeros(total_bins)
    for i in prange(total_bins):
        total_atoms = 0
        for j in prange(histogram.shape[0]):
            total_atoms += histogram[j][i]
        histogram_new[i] = total_atoms / settings.n_gr / nparticles

    # calculate the n(b) idela gas
    histogram_ideal = np.zeros(total_bins)
    rho = settings.rho / settings.sigma**3
    for i in prange(total_bins):
        histogram_ideal[i] = (
            4
            / 3
            * np.pi
            * rho
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
    # box_length_sigma = settings.xhi / settings.sigma / 2
    # other_line = np.sqrt(2) * box_length_sigma
    # plt.axvline(other_line, color="g", linestyle="--", label="sqrt(2) * box length")
    # plt.axvline(box_length_sigma, color="r", linestyle="--", label="box length/2")
    plt.title("Radial Distribution Function LJ Potential")
    plt.axhline(1, color="y", linestyle="--", label="Asymptote")
    plt.plot(x, rdf, label="g(r)")
    plt.xlabel("r/ sigma")
    plt.ylabel("g(r)")
    plt.legend()
    plt.savefig(os.path.join(settings.path, "g_r.png"))
    plt.show()
    # evaluate the plot
    # find max peak
    max_id = np.argmax(rdf)
    print("Max value in RDF is: ", rdf[max_id])
    print("The value is at position: ", max_id * bin_width / settings.sigma, " r/sigma")
    min_id_rel = np.argmin(rdf[max_id:])
    min_id = max_id + min_id_rel
    print("Next minimum after max is: ", rdf[min_id])
    print("The value is at position: ", min_id * bin_width / settings.sigma, " r/sigma")

    np.savetxt(os.path.join(settings.path, "g_r.txt"), rdf)
    np.savetxt(os.path.join(settings.path, "r.txt"), x * settings.sigma)


# 1d densities


@njit
def histogram_1d(x, xlo, xhi):
    rtot = xhi - xlo
    hist = np.zeros(int(rtot / settings.deltar))
    for xi in x:
        xi_mod = xi % rtot + xlo
        bin_n = int((xi_mod - xlo) / settings.deltar)
        hist[bin_n] += 1
    return hist


@njit
def calc_density_1d(histogram):
    # calculate the average number of atoms in each bin
    total_bins = histogram.shape[1]
    nparticles = settings.n1 * settings.n2 * settings.n3
    histogram_new = np.zeros(total_bins)
    for i in prange(total_bins):
        total_atoms = 0
        for j in prange(histogram.shape[0]):
            total_atoms += histogram[j][i]
        histogram_new[i] = total_atoms / settings.n_gr / nparticles * total_bins

    return histogram_new


def plot_density_1d(hist, direction, savename):
    x = np.arange(0, len(hist)) * settings.deltar / settings.sigma
    plt.plot(x, hist, label=rf"$\rho{direction}$")
    plt.xlabel(r"$r$ [$\sigma$]")
    plt.ylabel(rf"$\rho({direction})$")
    plt.legend()
    plt.savefig(os.path.join(settings.path, f"{savename}.png"))
    plt.show()
