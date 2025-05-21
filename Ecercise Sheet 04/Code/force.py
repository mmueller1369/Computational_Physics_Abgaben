import settings
import numpy as np
from numba import njit, prange


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
    #    for i in range(N-1):
    for i in prange(N - 1):
        j = i + 1
        #        for j in range(i+1,N):
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

    # LJ force
    for i in prange(N - 1):
        # LJ force
        for j in prange(i + 1, N):
            rijx = pbc(x[i], x[j], xlo, xhi)
            rijy = pbc(y[i], y[j], ylo, yhi)
            rijz = z[j] - z[i]

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

        # wall force
        rwall = ...
        if abs(rwall) < cutoff_wall:
            sf3 = sigma * sigma * sigma / (rwall * rwall * rwall)
            sf3a = abs(sf3)
            epot += ...

    return fx, fy, fz, epot
