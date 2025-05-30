from matplotlib import pyplot as plt
import os
import numpy as np
import settings

settings.init()

# ----------------- Part a ----------------- #
file_1 = np.loadtxt(os.path.join(settings.path, "temp_a_1.txt"))
step_1 = file_1[:, 0]
temp_1 = file_1[:, 1]
file_2 = np.loadtxt(os.path.join(settings.path, "temp_a_2.txt"))
step_2 = file_2[:, 0] + 40000
temp_2 = file_2[:, 1]

plt.plot(step_1, temp_1, label=r"$T_0=300 \,\mathrm{K}$")
plt.plot(step_2, temp_2, label=r"$T_0=100 \,\mathrm{K}$")
plt.legend()
plt.xlabel(r"$t~[\mathrm{fs}]$")
plt.ylabel(r"$T~[\mathrm{K}]$")
plt.savefig(os.path.join(settings.path, "a.png"), bbox_inches="tight")
plt.show()


# ----------------- Part c ----------------- #
file_temp = np.loadtxt(os.path.join(settings.path, "temp_cd.txt"))
step = file_temp[:, 0]
temp = file_temp[:, 1]
file_energy = np.loadtxt(os.path.join(settings.path, "energy_cd.txt"))
pe = file_energy[:, 1]
ke = file_energy[:, 2]
e = pe + ke
print(f"Mean Temperature: {np.mean(temp)}")
print(f"Mean Energy: {np.mean(e)}")
print(f"Variance Energy: {np.var(e)}")


# ----------------- Part d ----------------- #
file_rdf = np.loadtxt(os.path.join(settings.path, "rdf_cd.txt"))
r = file_rdf[:, 1]
rdf = file_rdf[:, 2]
