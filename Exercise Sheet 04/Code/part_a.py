import numpy as np
import settings
import os
import g_r
from matplotlib import pyplot as plt

settings.init()

histogram_x = np.loadtxt(os.path.join(settings.path, "histogram_x_a.txt"))
histogram_y = np.loadtxt(os.path.join(settings.path, "histogram_y_a.txt"))
histogram_z = np.loadtxt(os.path.join(settings.path, "histogram_z_a.txt"))

rho_x = g_r.calc_density_1d(histogram_x)
rho_y = g_r.calc_density_1d(histogram_y)
rho_z = g_r.calc_density_1d(histogram_z)

x = np.arange(0, len(rho_z)) * settings.deltar / settings.sigma
plt.plot(x, rho_z, label=r"$\rho(z)$")
plt.xlabel(r"$r$ [$\sigma$]")
plt.ylabel(r"$\rho$")
plt.legend()
plt.savefig(os.path.join(settings.path, "plot_a_z.png"))
plt.show()

x = np.arange(0, len(rho_x)) * settings.deltar / settings.sigma
plt.plot(x, rho_x, label=r"$\rho(x)$")
plt.plot(x, rho_y, label=r"$\rho(y)$")
plt.xlabel(r"$r$ [$\sigma$]")
plt.ylabel(r"$\rho$")
plt.legend()
plt.savefig(os.path.join(settings.path, "plot_a_xy.png"))
plt.show()

force_wall = np.loadtxt(os.path.join(settings.path, "force_wall_a.txt"))

mean_force = np.mean(np.abs(force_wall))
area = (settings.xhi - settings.xlo) * (settings.yhi - settings.ylo)
print(f"Mean pressure (g/mole * fs **2): {mean_force/area}")
