# module containing integration sheme:
"""
--------- units ---------
[r] = nm; [t] = fs; [epsilon] = kcal/mole; [m] = gram/mole;  [F] = (kcal/mole)/nm;
[T] = K; [v] = nm/fs; [k_B] = (kcal/mole)/K
conversion factor:
    from kcal*fs*fs/gram/nm to nm: 4.1868e-06
    from kcal*fs/gram/nm to nm/fs: 4.1868e-06
"""

import settings
import force
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def VelocityVerlet(
    x,
    y,
    z,
    vx,
    vy,
    vz,
    fx,
    fy,
    fz,
    xlo,
    xhi,
    ylo,
    yhi,
    zlo,
    zhi,
    eps,
    sigma,
    cutoff,
    deltat,
    mass,
):

    # conversion factor
    convdistance = 4.1868e-06
    convvelocity = 4.1868e-06
    fx0 = np.zeros(shape=len(x))
    fy0 = np.zeros(shape=len(y))
    fz0 = np.zeros(shape=len(z))
    N = len(x)
    dt = deltat
    # mass = mass

    # update the position at t+dt
    for i in prange(N):
        x[i] += vx[i] * dt + fx[i] * dt * dt * 0.5 / mass * convdistance
        y[i] += vy[i] * dt + fy[i] * dt * dt * 0.5 / mass * convdistance
        z[i] += vz[i] * dt + fz[i] * dt * dt * 0.5 / mass * convdistance

    # save the force at t
    fx0 = fx
    fy0 = fy
    fz0 = fz
    # update acceleration at t+dt
    fx, fy, fz, epot = force.forceLJ(
        x, y, z, xlo, xhi, ylo, yhi, zlo, zhi, eps, sigma, cutoff
    )

    # update the velocity
    for i in prange(N):
        vx[i] += 0.5 * dt * (fx[i] + fx0[i]) / mass * convvelocity
        vy[i] += 0.5 * dt * (fy[i] + fy0[i]) / mass * convvelocity
        vz[i] += 0.5 * dt * (fz[i] + fz0[i]) / mass * convvelocity

    return x, y, z, vx, vy, vz, fx, fy, fz, epot


def VelocityVerlet_wall_z(
    x,
    y,
    z,
    vx,
    vy,
    vz,
    fx,
    fy,
    fz,
    xlo,
    xhi,
    ylo,
    yhi,
    zlo,
    zhi,
    eps,
    sigma,
    cutoff,
    deltat,
    mass,
    eps_wall,
    sigma_wall,
    cutoff_wall,
):

    # conversion factor
    convdistance = 4.1868e-06
    convvelocity = 4.1868e-06
    fx0 = np.zeros(shape=len(x))
    fy0 = np.zeros(shape=len(y))
    fz0 = np.zeros(shape=len(z))
    N = len(x)
    dt = deltat
    # mass = mass

    # update the position at t+dt
    for i in prange(N):
        x[i] += vx[i] * dt + fx[i] * dt * dt * 0.5 / mass * convdistance
        y[i] += vy[i] * dt + fy[i] * dt * dt * 0.5 / mass * convdistance
        z[i] += vz[i] * dt + fz[i] * dt * dt * 0.5 / mass * convdistance

    # save the force at t
    fx0 = fx
    fy0 = fy
    fz0 = fz
    # update acceleration at t+dt
    fx, fy, fz, epot = force.forceLJ_wall_z(
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
    )

    # update the velocity
    for i in prange(N):
        vx[i] += 0.5 * dt * (fx[i] + fx0[i]) / mass * convvelocity
        vy[i] += 0.5 * dt * (fy[i] + fy0[i]) / mass * convvelocity
        vz[i] += 0.5 * dt * (fz[i] + fz0[i]) / mass * convvelocity

    return x, y, z, vx, vy, vz, fx, fy, fz, epot


@njit(parallel=True)
def KineticEnergy(vx, vy, vz, mass):

    ekin = 0
    N = len(vx)
    i = 0
    convvelocity = 4.1868e5

    for i in prange(N):
        ekin += (
            0.5 * mass * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]) * convvelocity
        )
    return ekin
