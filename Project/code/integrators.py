import forces
import numpy as np
from numba import njit, prange
import settings
# import settings_SI as settings
settings.init()


# @njit(parallel=True)
def VelocityVerlet(
    x, y, z,
    vx, vy, vz,
    fx, fy, fz,
    masses,
    dt,
    force, force_params
    ):

    fx0 = np.zeros(shape=len(x))
    fy0 = np.zeros(shape=len(y))
    fz0 = np.zeros(shape=len(z))
    N = len(x)

    # update the position at t+dt
    for i in prange(N):
        x[i] += vx[i]*dt + fx[i]/settings.conv_factor*dt*dt/2/masses[i]
        y[i] += vy[i]*dt + fy[i]/settings.conv_factor*dt*dt/2/masses[i]
        z[i] += vz[i]*dt + fz[i]/settings.conv_factor*dt*dt/2/masses[i]

    # save the force at t
    fx0 = fx
    fy0 = fy
    fz0 = fz
    # update acceleration at t+dt
    force_func = getattr(forces, f"force{force}")
    fx, fy, fz, energies = force_func(x, y, z, *force_params)
    
    # update the velocity
    for i in prange(N):
        vx[i] += dt/masses[i]/2 * (fx[i]+fx0[i])/settings.conv_factor
        vy[i] += dt/masses[i]/2 * (fy[i]+fy0[i])/settings.conv_factor
        vz[i] += dt/masses[i]/2 * (fz[i]+fz0[i])/settings.conv_factor

    return x, y, z, vx, vy, vz, fx, fy, fz, energies