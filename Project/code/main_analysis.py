import matplotlib.pyplot as plt
import os
import numpy as np
import math
import postprocessing
import settings
from scipy.optimize import curve_fit
settings.init()



# plt.plot(step, theta[:,0]*180/np.pi, label="angle")
# plt.show()

# plt.plot(step, si[:,0], label="si")
# plt.plot(step, sj[:,0], label="sj")
# plt.legend()
# plt.show()

name = "pert_b_0.1"
energyfile = os.path.join(settings.path, f"part_1/{name}_energy.txt")
energies = np.loadtxt(energyfile).T
plt.plot(energies[0]*0.5, energies[3], label = r"$E_{bond}$")
plt.plot(energies[0]*0.5, energies[4], label = r"$E_{angle}$")
plt.plot(energies[0]*0.5, energies[5], label = r"$E_{kin}$")
plt.plot(energies[0]*0.5, np.sum(energies[1:], axis=0), label = r"$E_{tot}$")
plt.title(f'Energies for file {name}')
plt.xlabel(r"$t$ [fs]")
plt.ylabel(r"$E$ [kcal/mole]")
plt.legend()
plt.show()

# def oscillation(t, A, omega, phi, offset):
#     return A * np.cos(omega * t + phi) + offset

# test for si
# name = "long_pert_s_0.1"
# trajfile = os.path.join(settings.path, f"part_1/{name}_traj.txt")
# energyfile = os.path.join(settings.path, f"part_1/{name}_energy.txt")
# step, x, y, z = postprocessing.read_data(trajfile)
# si, sj, theta = postprocessing.calculate_molecule_properties(x, y, z)

# # Conversion factor from (kcal/mol)/(u*nm^2) to 1/fs (angular frequency)
# conv_omega = math.sqrt(4184 * 1e21) * 1e-15
# omega_exp = math.sqrt(settings.k_bond / settings.masses[0])/conv_omega
# print(conv_omega, omega_exp)
# # params, _ = curve_fit(oscillation, step*settings.deltat, sj[:,0])#, p0=[0.01, omega_exp, 0, settings.s0])
# # print("Fitted parameters:")
# # print(f"A: {params[0]}, omega: {params[1]}, phi: {params[2]}, offset: {params[3]}")
# plt.figure(figsize=(10, 5))
# plt.plot(step*settings.deltat, sj[:,0], label="si")
# # plt.plot(step*settings.deltat, oscillation(step*settings.deltat, *params), label="fitted curve")
# plt.xlabel(r"$t$ [fs]")
# plt.ylabel(r"$s_i$ [nm]")
# plt.legend()
# plt.show()

