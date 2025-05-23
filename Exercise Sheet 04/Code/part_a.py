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

plt.plot(rho_x)
plt.show()

g_r.plot_density_1d(rho_x, "x")
g_r.plot_density_1d(rho_y, "y")
g_r.plot_density_1d(rho_z, "z")
