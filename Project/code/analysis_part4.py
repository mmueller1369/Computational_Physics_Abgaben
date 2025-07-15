import matplotlib.pyplot as plt
import os
import numpy as np
import math
from scipy.optimize import curve_fit
from postprocessing import PostprocessingTools, fermi_distribution
import settings
settings.init()


# ausrichtung by distance
# overall asphericity
# molecules in droplet over time -> freeze
# droplet diameter scaling with nparticles



trajfileeq = os.path.join(settings.path, f"part_3/traj_eq_long.txt")
trajfile = os.path.join(settings.path, f"part_3/traj_long.txt")
# eq = PostprocessingTools(trajfileeq, 1001)
prod = PostprocessingTools(trajfile, 100)

r, rho, params, pcov = prod.calculate_rho()
rH, rhoH, paramsH, pcovH = prod.calculate_rho(only_hatoms=True)
_, rgs, asphericities, _, _ = prod.calculate_droplet_properties()
mean_rg = np.mean(rgs)
uncertainty_rg = np.std(rgs)
mean_b = np.mean(asphericities)
uncertainty_b = np.std(asphericities)

print(f"Radius = {params[0]} +- {np.sqrt(pcov[0][0])}\n \
        Bulk density = {params[1]} +- {np.sqrt(pcov[1][1])}\n \
        Sharpness = {params[2]} +- {np.sqrt(pcov[2][2])}\n \
        Radius H = {paramsH[0]} +- {np.sqrt(pcovH[0][0])}\n \
        Bulk density H / 2 = {paramsH[1]/2} +- {np.sqrt(pcovH[1][1])/2}\n \
        Sharpness H = {paramsH[2]} +- {np.sqrt(pcovH[2][2])}\n \
        Radius of Gyration = {mean_rg} +- {uncertainty_rg}\n \
        Asphericity = {mean_b} +- {uncertainty_b}")
plt.plot(r[1:], rho[1:], label='data')
plt.plot(rH[1:], rhoH[1:]/2, label='data H')
plt.plot(r[1:], fermi_distribution(r[1:], *params), label='fit', color='red')
# plt.axvline(mean_rg, label=r'R_g\pm\Delta', color='black')
# plt.axvline(mean_rg+uncertainty_b, ls="dashed", color='black')
# plt.axvline(mean_rg-uncertainty_b, ls="dashed", color='black')
plt.legend()
plt.xlabel(r'$r$ [nm]')
plt.ylabel(r'$\bar\rho$ [particles/nm$^3$]')
plt.show()


dist, value = prod.calculate_dipole_projections_by_distance()
fig, ax1 = plt.subplots()
color1 = 'blue'
ax1.set_xlabel(r'$r$ [nm]')
ax1.set_ylabel('Dipole projection', color=color1)
ln1 = ax1.plot(dist[1:], value[1:], label='dipole projection', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'red'
ax2.set_ylabel(r'$\bar\rho$ [particles/nm$^3$]', color=color2)
ln2 = ax2.plot(r[1:], rho[1:], label='density', color='orange')
ln2 = ax2.plot(rH[1:], rhoH[1:]/2, label='H density', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# Legende für beide Achsen kombinieren
lns = ln1 + ln2
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='best')

plt.show()


dist, energy, field = prod.calculate_electrostatic_potential_and_field(dr_hist=settings.dr_hist)
plt.plot(dist, energy)
plt.xlabel(r'$r$ [nm]')
plt.ylabel(r'$E$ [kcal/mol]')
plt.show()

plt.plot(dist, field)
plt.xlabel(r'$r$ [nm]')
plt.ylabel(r'$F$ [kcal/mol/nm]')
plt.show()