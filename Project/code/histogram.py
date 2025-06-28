import math
import numpy as np
import settings
from numba import njit, prange


@njit
def create_histogram(x, y, z):
    # Calculate distances between COM of O
    com_x = np.zeros(settings.nmol)
    com_y = np.zeros(settings.nmol)
    com_z = np.zeros(settings.nmol)
    for i in prange(settings.nmol):
        id_O = 3*i
        com_x[i] = 0.5 * (x[id_O] + x[id_O + 1])
        com_y[i] = 0.5 * (y[id_O] + y[id_O + 1])
        com_z[i] = 0.5 * (z[id_O] + z[id_O + 1])
    hist = np.zeros(int(settings.rmax_hist / settings.dr_hist))
    for i in prange(settings.nmol):
        for j in prange(i + 1, settings.nmol):
            rijx = com_x[j] - com_x[i]
            rijy = com_y[j] - com_y[i]
            rijz = com_z[j] - com_z[i]
            r2 = rijx * rijx + rijy * rijy + rijz * rijz
            if r2 < settings.rmax_hist * settings.rmax_hist:
                r = math.sqrt(r2)
                bin_n = r//settings.dr_hist
                hist[bin_n] += 2
    
    return hist


@njit
def calc_RDF(histograms):
    # calculate the average number of atoms in each bin
    n_gr = histograms.shape[1]
    histogram_new = np.zeros(n_gr)
    for i in prange(n_gr):
        total_atoms = 0
        for j in prange(histograms.shape[0]):
            total_atoms += histograms[j][i]
        histogram_new[i] = total_atoms / n_gr / settings.nmol

    # calculate the n(b) ideal gas
    histogram_ideal = np.zeros(n_gr)
    rho = settings.rho / settings.sigma**3
    for i in prange(n_gr):
        r = i*settings.dr_hist
        histogram_ideal[i] = 4/3*np.pi*rho * ((r + settings.dr_hist)**3 - r**3)

    return histogram_new / histogram_ideal, [histogram_new, histogram_ideal]