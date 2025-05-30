from matplotlib import pyplot as plt
import os
import numpy as np
import settings

settings.init()

# # ----------------- Part a ----------------- #
# file_1 = np.loadtxt(os.path.join(settings.path, "temp_a_1.txt"))
# step_1 = file_1[:, 0]
# temp_1 = file_1[:, 1]
# file_2 = np.loadtxt(os.path.join(settings.path, "temp_a_2.txt"))
# step_2 = file_2[:, 0] + 40000
# temp_2 = file_2[:, 1]

# plt.plot(step_1, temp_1, label=r"$T_0=300 \,\mathrm{K}$")
# plt.plot(step_2, temp_2, label=r"$T_0=100 \,\mathrm{K}$")
# plt.legend()
# plt.xlabel(r"$t~[\mathrm{fs}]$")
# plt.ylabel(r"$T~[\mathrm{K}]$")
# plt.savefig(os.path.join(settings.path, "a.png"), bbox_inches="tight")
# plt.show()


# ----------------- Part c ----------------- #
file_temp = np.loadtxt(os.path.join(settings.path, "temp_cd.txt"))
step = file_temp[:, 0]
temp = file_temp[:, 1]
file_energy = np.loadtxt(os.path.join(settings.path, "energy_cd.txt"))
pe = file_energy[:, 1]
ke = file_energy[:, 2]
e = pe + ke
print(f"Part c: Mean Temperature: {np.mean(temp)}")
print(f"Part c: Mean Energy: {np.mean(e)}")
print(f"Part c: Variance Energy: {np.var(e)}")
print(f"Part c: Mean Potential Energy: {np.mean(pe)}")
print(f"Part c: Variance Potential Energy: {np.var(pe)}")
print(f"Part c: Mean Kinetic Energy: {np.mean(ke)}")
print(f"Part c: Variance Kinetic Energy: {np.var(ke)}")


# ----------------- Part d ----------------- #
def potentialLJ(r):
    sf6a = (settings.sigma / settings.cutoff) ** 6
    epotcut = 4.0 * settings.eps * sf6a * (sf6a - 1.0)
    mask = (r < settings.cutoff) & (r > 0)
    sf6 = np.zeros_like(r)
    sf6[mask] = (settings.sigma / r[mask]) ** 6
    epot = np.zeros_like(r)
    epot[mask] = 4.0 * settings.eps * sf6[mask] * (sf6[mask] - 1.0) - epotcut
    return epot


file_rdf = np.loadtxt(os.path.join(settings.path, "rdf_cd.txt"))
r = file_rdf[:, 0]
g_r = file_rdf[:, 1]
pot_r = potentialLJ(r * settings.sigma)

integral_core = g_r * pot_r * 4 * np.pi * r**2
rho = settings.rho / settings.sigma**3
integral = rho / 2 * np.sum(integral_core) * settings.deltar
emean_gr = integral * settings.n1 * settings.n2 * settings.n3
print(f"Part d: Mean Energy via g_r: {emean_gr}")

fig, ax1 = plt.subplots()
ax1.plot(r, g_r, color="tab:blue")
ax1.set_xlabel(r"$r / \sigma$")
ax1.set_ylabel(r"$g(r)$", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(r, integral_core, color="tab:red")
ax2.set_ylabel("integral core", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

fig.tight_layout()
plt.savefig(os.path.join(settings.path, "d.png"), bbox_inches="tight")
plt.show()
