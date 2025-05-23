import numpy as np
import settings
import os
import g_r
from matplotlib import pyplot as plt

settings.init()

histogram_x = np.loadtxt(os.path.join(settings.path, "histogram_x.txt"))
histogram_y = np.loadtxt(os.path.join(settings.path, "histogram_y.txt"))
histogram_z = np.loadtxt(os.path.join(settings.path, "histogram_z.txt"))

rho_x = g_r.calc_density_1d(histogram_x)
rho_y = g_r.calc_density_1d(histogram_y)
rho_z = g_r.calc_density_1d(histogram_z)

# g_r.plot_density_1d(rho_x, "x")
# g_r.plot_density_1d(rho_y, "y")
# g_r.plot_density_1d(rho_z, "z")

force_wall = np.loadtxt(os.path.join(settings.path, "force_wall.txt"))

mean_force = np.mean(np.abs(force_wall))
area = (settings.xhi - settings.xlo) * (settings.yhi - settings.ylo)
print(f"Mean pressure (g/mole * fs **2): {mean_force/area}")
