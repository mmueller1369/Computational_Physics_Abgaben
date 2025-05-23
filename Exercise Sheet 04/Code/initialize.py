import settings
import random
import math
import debug
import numpy as np
from tqdm import tqdm


def InitializeAtoms():

    nx = 0
    ny = 0
    ny = 0
    n = 0
    x = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
    y = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
    z = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
    vx = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
    vy = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
    vz = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
    nx = 0
    pbar = tqdm(total=settings.n1, desc="Initializing atoms")
    while nx < settings.n1:
        ny = 0
        while ny < settings.n2:
            nz = 0
            while nz < settings.n3:
                x0 = nx * settings.deltaxyz + settings.deltaxyz / 2.0
                y0 = ny * settings.deltaxyz + settings.deltaxyz / 2.0
                z0 = nz * settings.deltaxyz + settings.deltaxyz / 2.0

                vx0 = 0.5 - random.randint(0, 1)
                vy0 = 0.5 - random.randint(0, 1)
                vz0 = 0.5 - random.randint(0, 1)

                x[n] = x0
                y[n] = y0
                z[n] = z0

                vx[n] = vx0
                vy[n] = vy0
                vz[n] = vz0

                n += 1
                nz += 1

            ny += 1

        nx += 1
        pbar.update(1)
    pbar.close()
    settings.nparticles = n

    # cancel the linear momentum
    svx = np.sum(vx)
    svy = np.sum(vy)
    svz = np.sum(vz)

    vx -= svx / settings.nparticles
    vy -= svy / settings.nparticles
    vz -= svz / settings.nparticles

    # rescale the velocity to the desired temperature
    Trandom = temperature(vx, vy, vz)
    vx, vy, vz = rescalevelocity(vx, vy, vz, settings.Tdesired, Trandom)

    # cancel the linear momentum
    svx = np.sum(vx)
    svy = np.sum(vy)
    svz = np.sum(vz)

    vx -= svx / settings.nparticles
    vy -= svy / settings.nparticles
    vz -= svz / settings.nparticles

    return x, y, z, vx, vy, vz


def temperature(vx, vy, vz):
    # convunits is the conversion factor
    convunits = 238845.9  # from (gram/mole)*(nm/fs)^2/((kcal/mole)/K) to K
    vsq = 0.0
    vsq = np.sum(np.multiply(vx, vx) + np.multiply(vy, vy) + np.multiply(vz, vz))
    return settings.mass * vsq / 2.0 / settings.kb / settings.nparticles * convunits


def rescalevelocity(vx, vy, vz, T1, T2):

    vx = vx * math.sqrt(T1 / T2)
    vy = vy * math.sqrt(T1 / T2)
    vz = vz * math.sqrt(T1 / T2)
    return vx, vy, vz


def histogram():
    n_bins = int(settings.rmax / settings.deltar)
    # dann folgt damit für die bin_breite
    width_bin = settings.rmax / n_bins
    return np.zeros((settings.n_gr, n_bins)), width_bin


def histogram_1d(xhi, xlo):
    rtot = xhi - xlo
    n_bins = int(rtot / settings.deltar)
    width_bin = rtot / n_bins
    return np.zeros((settings.n_gr, n_bins)), width_bin


def block_histograms(zhi, zlo):
    rtot = zhi - zlo
    n_bins = int(rtot / settings.deltar)
    width_bin = rtot / n_bins
    return np.zeros((settings.block_length, n_bins)), width_bin
