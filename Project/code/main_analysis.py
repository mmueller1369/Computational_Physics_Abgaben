import matplotlib.pyplot as plt
import os
import numpy as np
import postprocessing
import settings
settings.init()

name = "pert_b_0.1"
trajfile = os.path.join(settings.path, f"part_1/{name}_traj.txt")
energyfile = os.path.join(settings.path, f"part_1/{name}_energy.txt")
step, x, y, z = postprocessing.read_pos(trajfile)
si, sj, theta = postprocessing.calculate_molecule_properties(x, y, z)

# plt.plot(step, theta[:,0]*180/np.pi, label="angle")
# plt.show()

# plt.plot(step, si[:,0], label="si")
# plt.plot(step, sj[:,0], label="sj")
# plt.legend()
# plt.show()

energies = np.loadtxt(energyfile).T
plt.plot(energies[0], energies[3], label = "e_bond")
plt.plot(energies[0], energies[4], label = "e_angle")
plt.plot(energies[0], energies[5], label = "e_kin")
plt.plot(energies[0], np.sum(energies[1:], axis=0), label = "e_tot")
plt.legend()
plt.show()
