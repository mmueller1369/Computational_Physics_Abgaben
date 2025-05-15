import settings
import math
from force import pbc
from numba import njit, prange
import numpy as np


@njit(parallel=True)
def hist_g_r(x, y, z, xlo, xhi, ylo, yhi, zlo, zhi):
    """
    Calculate the g(r) function for a given set of coordinates.
    """
    dr = settings.dr * settings.sigma
    N = len(x)
    rmax = (
        max((xhi - xlo), (yhi - ylo), (zhi - zlo)) / 2
    )  # maximum distance possible according to minimum image convention
    nbins = int(rmax / dr)
    hist = np.zeros(nbins)

    for i in prange(N - 1):
        j = i + 1
        for j in prange(i + 1, N):
            rijx = pbc(x[i], x[j], xlo, xhi)
            rijy = pbc(y[i], y[j], ylo, yhi)
            rijz = pbc(z[i], z[j], zlo, zhi)

            r2 = rijx * rijx + rijy * rijy + rijz * rijz
            r = math.sqrt(r2)

            bin_index = int(r / dr)
            hist[bin_index] += 1

    return hist


def g_r(hists):
    """
    Calculate the g(r) function from the histograms.
    """
    dr = settings.dr * settings.sigma
    N = settings.n1 * settings.n2 * settings.n3
    N_gr = len(hists)
    bins = np.arange(0, N_gr * dr, dr)
    n_b = np.sum(hists, axis=0) / (N_gr * N)
    n_id = 4 / 3 * math.pi * settings.rho * ((bins * dr + dr) ** 3 - (bins * dr) ** 3)
    g_r = n_b / n_id
    distance = bins - dr / 2
    return g_r, distance
