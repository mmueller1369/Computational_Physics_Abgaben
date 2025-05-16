import numpy as np
import settings
import os
import matplotlib.pyplot as plt

settings.init()


def potentialLJ(r):
    sf6a = (settings.sigma / settings.cutoff) ** 6
    epotcut = 8.0 * settings.eps * sf6a * (sf6a - 1.0)
    sf6 = np.where(r < settings.cutoff, (settings.sigma / r) ** 6, 0.0)
    epot = np.where(
        r < settings.cutoff,
        8.0 * settings.eps * sf6 * (sf6 - 1.0) - epotcut,
        0,
    )
    epot[0] = 0.0
    return epot


### Integral
r = np.loadtxt(os.path.join(settings.path, "r.txt"))  # in nm
# r = np.linspace(0.15, 1, 100)
g_r = np.loadtxt(os.path.join(settings.path, "g_r.txt"))
pot_r = potentialLJ(r)  # in kcal/mole

integral_core = (
    g_r * pot_r * 4 * np.pi * r**2
)  # over 3d space thus the volume element of the sphere
rho = settings.rho / settings.sigma**3  # in nm^-3
integral = rho / 2 * np.sum(integral_core) * settings.deltar
print(integral * settings.n1 * settings.n2 * settings.n3)


# integral = rho / 2 * np.sum(integral_core[2:] * (settings.deltar / settings.sigma))
# print(r, pot_r)
# print(integral)
# plt.plot(r, pot_r)
# plt.plot(r, g_r)
# plt.ylim(-1, 1)
# plt.show()
