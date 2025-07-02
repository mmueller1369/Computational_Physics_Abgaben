import settings
import random
import math
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


def InitializeAtomsBond():
    # Moleküle auf Gitter verteilen, jeweils 2 Atome pro Molekül
    nmol = settings.n1 * settings.n2 * settings.n3
    natoms = 2 * nmol
    x = np.zeros(natoms)
    y = np.zeros(natoms)
    z = np.zeros(natoms)
    vx = np.zeros(natoms)
    vy = np.zeros(natoms)
    vz = np.zeros(natoms)
    n = 0
    pbar = tqdm(total=settings.n1, desc="Initializing molecules")
    for nx in range(settings.n1):
        for ny in range(settings.n2):
            for nz in range(settings.n3):
                # Mittelpunkt des Moleküls
                x0 = nx * settings.deltaxyz + settings.deltaxyz / 2.0
                y0 = ny * settings.deltaxyz + settings.deltaxyz / 2.0
                z0 = nz * settings.deltaxyz + settings.deltaxyz / 2.0
                # zufällige Richtung für Bindungsvektor
                theta = np.pi * random.random()
                phi = 2 * np.pi * random.random()
                dx = settings.b0 / 2 * np.sin(theta) * np.cos(phi)
                dy = settings.b0 / 2 * np.sin(theta) * np.sin(phi)
                dz = settings.b0 / 2 * np.cos(theta)
                # Positionen der beiden Atome
                x[n] = x0 - dx
                y[n] = y0 - dy
                z[n] = z0 - dz
                x[n+1] = x0 + dx
                y[n+1] = y0 + dy
                z[n+1] = z0 + dz
                # gleiche Anfangsgeschwindigkeit für beide Atome
                vx0 = 0.5 - random.randint(0, 1)
                vy0 = 0.5 - random.randint(0, 1)
                vz0 = 0.5 - random.randint(0, 1)
                vx[n] = vx0
                vy[n] = vy0
                vz[n] = vz0
                vx[n+1] = vx0
                vy[n+1] = vy0
                vz[n+1] = vz0
                n += 2
        pbar.update(1)
    pbar.close()
    settings.nparticles = natoms
    # Impuls nullen
    svx = np.sum(vx)
    svy = np.sum(vy)
    svz = np.sum(vz)
    vx -= svx / natoms
    vy -= svy / natoms
    vz -= svz / natoms
    # Temperatur reskalieren
    Trandom = temperature(vx, vy, vz)
    vx, vy, vz = rescalevelocity(vx, vy, vz, settings.Tdesired, Trandom)
    # Impuls erneut nullen
    svx = np.sum(vx)
    svy = np.sum(vy)
    svz = np.sum(vz)
    vx -= svx / natoms
    vy -= svy / natoms
    vz -= svz / natoms
    return x, y, z, vx, vy, vz


def temperature(vx, vy, vz):
    # convunits is the conversion factor
    convunits = 238845.9  # from (gram/mole)*(nm/fs)^2/((kcal/mole)/K) to K
    vsq = 0.0
    vsq = np.sum(np.multiply(vx, vx) + np.multiply(vy, vy) + np.multiply(vz, vz))
    # EVTL DURCH 3 TEILEN?
    return settings.mass * vsq / 2.0 / settings.kb / settings.nparticles * convunits


def rescalevelocity(vx, vy, vz, T1, T2):

    vx = vx * math.sqrt(T1 / T2)
    vy = vy * math.sqrt(T1 / T2)
    vz = vz * math.sqrt(T1 / T2)
    return vx, vy, vz


def berendsen_thermostat(vx, vy, vz, T1, T2, tau, dt):

    multiplier = math.sqrt(1 + (dt / tau) * ((T1 / T2) - 1))
    vx = vx * multiplier
    vy = vy * multiplier
    vz = vz * multiplier
    return vx, vy, vz


def andersen_thermostat(vx, vy, vz, T0, Tsystem, nu, dt):
    # Tsystem just needed for generalization ("Guter Pfusch ist keine schlechte Arbeit")
    convunits = 238845.9 * 3 / 2
    variance = np.sqrt(settings.kb * T0 / settings.mass / convunits)
    for i, _ in enumerate(vx):
        if np.random.rand() < nu * dt:
            vx[i] = np.random.normal(0, variance)
            vy[i] = np.random.normal(0, variance)
            vz[i] = np.random.normal(0, variance)
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
