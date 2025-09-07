import math
from numba import njit, prange
import settings
settings.init()


@njit(parallel=True)
def computeKineticEnergy(vx, vy, vz, masses):
    ekin = 0
    for i in prange(len(vx)):
        ekin += 0.5 * masses[i] * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i])
    return ekin*settings.conv_factor


def computeTemperature(vx, vy, vz, masses):
    ekin = computeKineticEnergy(vx, vy, vz, masses)
    ekin_mean = ekin/len(masses)
    temp = 2/3 * ekin_mean / settings.kb
    return temp


def rescaleVelocity(vx, vy, vz, Tdesired, Tnow):
    multiplier = math.sqrt(Tdesired / Tnow)
    vx = vx * multiplier
    vy = vy * multiplier
    vz = vz * multiplier
    return vx, vy, vz


def BerendsenThermostat(vx, vy, vz, Tnow, Tdesired, tau, dt):
    multiplier = math.sqrt(1 + (dt / tau) * ((Tdesired / Tnow) - 1))
    vx = vx * multiplier
    vy = vy * multiplier
    vz = vz * multiplier
    return vx, vy, vz