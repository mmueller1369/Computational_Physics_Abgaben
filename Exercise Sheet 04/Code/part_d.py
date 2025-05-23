import numpy as np
import settings
import os
import g_r
from matplotlib import pyplot as plt

settings.init()

histogram_1 = np.loadtxt(os.path.join(settings.path, "histogram_z_d1.txt"))
histogram_2 = np.loadtxt(os.path.join(settings.path, "histogram_z_d2.txt"))
histogram_3 = np.loadtxt(os.path.join(settings.path, "histogram_z_d3.txt"))

rho_1 = g_r.calc_density_1d(histogram_1)
rho_2 = g_r.calc_density_1d(histogram_2)
rho_3 = g_r.calc_density_1d(histogram_3)

x = np.arange(0, len(rho_1)) * settings.deltar / settings.sigma

plt.plot(x, rho_3, label=r"$k=100$")
plt.plot(x, rho_2, label=r"$k=10$")
plt.plot(x, rho_1, label=r"$k=1$")

plt.xlabel(r"$r$ [$\sigma$]")
plt.ylabel(r"$\rho(z)$")
plt.legend()
plt.savefig(os.path.join(settings.path, "plot_d.png"))
plt.show()
