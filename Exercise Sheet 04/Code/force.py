import settings
import numpy as np
from numba import njit, prange
import math


#### unit of the force: (kcal/mole)/nm


@njit(parallel=True)
def forceLJ(x, y, z, xlo, xhi, ylo, yhi, zlo, zhi, eps, sigma, cutoff):

    fx = np.zeros(shape=len(x))
    fy = np.zeros(shape=len(x))
    fz = np.zeros(shape=len(x))
    N = len(x)

    i = 0
    sf2a = sigma * sigma / cutoff / cutoff
    sf6a = sf2a * sf2a * sf2a

    epotcut = 4.0 * eps * sf6a * (sf6a - 1.0)
    epot = 0

    for i in prange(N - 1):
        j = i + 1
        for j in prange(i + 1, N):
            rijx = pbc(x[i], x[j], xlo, xhi)
            rijy = pbc(y[i], y[j], ylo, yhi)
            rijz = pbc(z[i], z[j], zlo, zhi)

            r2 = rijx * rijx + rijy * rijy + rijz * rijz
            # calculate fx, fy, fz
            if r2 < cutoff * cutoff:
                sf2 = sigma * sigma / r2
                sf6 = sf2 * sf2 * sf2
                epot += 4.0 * eps * sf6 * (sf6 - 1.0) - epotcut
                ff = 24.0 * eps * sf6 * (sf6 - 0.5) / r2
                fx[i] -= ff * rijx
                fy[i] -= ff * rijy
                fz[i] -= ff * rijz

                fx[j] += ff * rijx
                fy[j] += ff * rijy
                fz[j] += ff * rijz

    return fx, fy, fz, epot


@njit
def pbc(xi, xj, xlo, xhi):

    l = xhi - xlo

    xi = xi % l
    xj = xj % l

    rij = xj - xi
    if abs(rij) > 0.5 * l:
        rij = rij - np.sign(rij) * l

    return rij


@njit(parallel=True)
def forceLJ_wall_z(
    x,
    y,
    z,
    xlo,
    xhi,
    ylo,
    yhi,
    zlo,
    zhi,
    eps,
    sigma,
    cutoff,
    eps_wall,
    sigma_wall,
    cutoff_wall,
):

    fx = np.zeros(shape=len(x))
    fy = np.zeros(shape=len(x))
    fz = np.zeros(shape=len(x))
    N = len(x)

    i = 0
    sf2a = sigma * sigma / cutoff / cutoff
    sf6a = sf2a * sf2a * sf2a

    epotcut = 4.0 * eps * sf6a * (sf6a - 1.0)
    epot = 0
    lz = zhi - zlo

    for i in prange(N - 1):
        # LJ force
        for j in prange(i + 1, N):
            rijx = pbc(x[i], x[j], xlo, xhi)
            rijy = pbc(y[i], y[j], ylo, yhi)
            rijz = z[j] - z[i]

            r2 = rijx * rijx + rijy * rijy + rijz * rijz
            if r2 < cutoff * cutoff:
                sf2 = sigma * sigma / r2
                sf6 = sf2 * sf2 * sf2
                epot += 4.0 * eps * sf6 * (sf6 - 1.0) - epotcut
                ff = 24.0 * eps * sf6 * (sf6 - 0.5) / r2
                fx[i] -= ff * rijx
                fy[i] -= ff * rijy
                fz[i] -= ff * rijz

                fx[j] += ff * rijx
                fy[j] += ff * rijy
                fz[j] += ff * rijz

        # wall force
        # calculate the closest distance to the wall (including the sign for the direction of the force)
        zi = z[i]
        r_wall = zi - (zlo + lz * (zi > lz / 2 + zlo))
        if abs(r_wall) < cutoff_wall:
            sf1 = sigma_wall / r_wall
            sf3 = sf1**3
            sf3a = abs(sf3)
            epot += 3.0 * math.sqrt(3.0) / 2 * eps_wall * sf3a * (sf3a**2 - 1.0)
            ff = (
                9.0
                * math.sqrt(3.0)
                / 2
                * eps_wall
                * sf3a
                * (3 * sf3a**2 - 1.0)
                / r_wall**2
            )
            fz[i] += ff * r_wall

    # for the last particle which isn't included in the loop
    zi = z[-1]
    r_wall = zi - (zlo + lz * (zi > lz / 2 + zlo))
    if abs(r_wall) < cutoff_wall:
        sf1 = sigma_wall / r_wall
        sf3 = sf1**3
        sf3a = abs(sf3)
        epot += 3.0 * math.sqrt(3.0) / 2 * eps_wall * sf3a * (sf3a**2 - 1.0)
        ff = (
            9.0 * math.sqrt(3.0) / 2 * eps_wall * sf3a * (3 * sf3a**2 - 1.0) / r_wall**2
        )
        fz[-1] += ff * r_wall

    return fx, fy, fz, epot


@njit(parallel=True)
def measure_force_wall(z, zlo, zhi, eps_wall, sigma_wall, cutoff_wall):
    # calculate the closest distance to the wall (including the sign for the direction of the force)
    force_wall = np.zeros(shape=len(z))
    lz = zhi - zlo
    rwalls = np.zeros(shape=len(z))
    for i in prange(len(z)):
        zi = z[i]
        r_wall = zi - (zlo + lz * (zi > lz / 2 + zlo))
        rwalls[i] = r_wall
        # wall force
        if abs(r_wall) < cutoff_wall:
            sf1 = sigma_wall / r_wall
            sf3 = sf1**3
            sf3a = abs(sf3)
            force_wall[i] = (
                9.0
                * math.sqrt(3.0)
                / 2
                * eps_wall
                * sf3a
                * (3 * sf3a**2 - 1.0)
                / r_wall
            )
    return force_wall, rwalls
