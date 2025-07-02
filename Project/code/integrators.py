import forces
import numpy as np
from numba import njit, prange
import settings
settings.init()


@njit(parallel=True)
def VelocityVerlet(
    x, y, z,
    vx, vy, vz,
    fx, fy, fz,
    dt,
    k_bond, s0, k_angle, theta0, # Intramolecular parameters
    eps, sigma, cutoff, qO, qH, eps0_el, alpha, # Intermolecular parameters
    ):

    fx0 = np.zeros(shape=len(x))
    fy0 = np.zeros(shape=len(y))
    fz0 = np.zeros(shape=len(z))
    N = len(x)

    # update the position at t+dt
    for i in prange(N):
        x[i] += vx[i]*dt + fx[i]/settings.conv_factor*dt*dt/2/settings.masses[i]
        y[i] += vy[i]*dt + fy[i]/settings.conv_factor*dt*dt/2/settings.masses[i]
        z[i] += vz[i]*dt + fz[i]/settings.conv_factor*dt*dt/2/settings.masses[i]

    # save the force at t
    fx0 = fx
    fy0 = fy
    fz0 = fz
    # update acceleration at t+dt
    fx, fy, fz, energies = forces.forceH2O(x, y, z,
                                    k_bond, s0, k_angle, theta0,
                                    eps, sigma, cutoff, qO, qH, eps0_el, alpha)
    
    # update the velocity
    for i in prange(N):
        vx[i] += dt/settings.masses[i]/2 * (fx[i]+fx0[i])/settings.conv_factor
        vy[i] += dt/settings.masses[i]/2 * (fy[i]+fy0[i])/settings.conv_factor
        vz[i] += dt/settings.masses[i]/2 * (fz[i]+fz0[i])/settings.conv_factor

    return x, y, z, vx, vy, vz, fx, fy, fz, energies