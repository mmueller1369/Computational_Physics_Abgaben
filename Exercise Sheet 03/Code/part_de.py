import numpy as np
import settings
import os

settings.init()


def potentialLJ(r):
    sf6a = (settings.sigma / settings.cutoff) ** 6
    epotcut = 8.0 * settings.eps * sf6a * (sf6a - 1.0)
    sf6 = r**6
    epot = 8.0 * settings.eps * sf6 * (sf6 - 1.0) - epotcut
    return epot


### Integral
rmin = 0.0
r = np.loadtxt(os.path.join(settings.path, "r.txt"))
g_r = np.loadtxt(os.path.join(settings.path, "g_r.txt"))
pot_r = potentialLJ(r)
integral = settings.rho / 2 * np.sum((g_r * pot_r / settings.rmax))
print(integral)
