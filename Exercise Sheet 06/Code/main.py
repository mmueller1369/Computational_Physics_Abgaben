import execute
import settings
import initialize
import numpy as np
import matplotlib.pyplot as plt
import os

settings.init()
x, y, z, vx, vy, vz = initialize.InitializeAtoms()
f_initial = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]


# ----------------- Part a ----------------- #
def B2(beta, eps, sigma):
    def potentialLJ(r, eps, sigma):
        mask = r > 0
        sf6 = np.zeros_like(r)
        sf6[mask] = (sigma / r[mask]) ** 6
        epot = np.zeros_like(r)
        epot[mask] = 4.0 * eps * sf6[mask] * (sf6[mask] - 1.0)
        return epot

    cutoff = (settings.xhi - settings.xlo) * 3
    deltar = cutoff / 10000
    r = np.arange(0, cutoff, deltar)
    U = potentialLJ(r, eps, sigma)
    integral_core = np.exp(-beta * U) - 1
    integral_core *= 4 * np.pi * r**2  # conversion into spherical coordinates
    integral = -1 / 2 * np.sum(integral_core) * deltar
    return integral


beta = 1 / (settings.Tdesired * settings.kb)
sigma = settings.sigma
print(
    f"a: for our parameter set (argon): B2/eps = {B2(
        beta,
        1,  # set to unity
        sigma,
    ):.4f}"
)

epss = np.linspace(0, 1, 1000) / beta
B2_eps = [B2(beta, eps, sigma) / sigma**3 for eps in epss]

plt.figure(figsize=(6, 4))
plt.plot(epss * beta, B2_eps, label=r"$B_2/\sigma^3$", color="b")
plt.axvline(settings.eps * beta, label=r"$\epsilon\beta_\mathrm{Argon}$", color="r")
plt.xlabel(r"$\epsilon\beta$")
plt.ylabel(r"$B_2/\sigma^3$")
plt.grid()
plt.legend()
plt.savefig(os.path.join(settings.path, "a.png"), bbox_inches="tight", dpi=300)
plt.show()


# ----------------- Part b ----------------- #
equilibrated_config = execute.run_simulation(
    initial_config=initial_config,
    integrator="VelocityVerlet",
    force="LJ",
    steps=settings.nsteps_equi,
    thermostat="andersen_thermostat",
    thermostat_params=[settings.Tdesired, settings.nu, settings.deltat],
    n_thermostat=1,
    trajfile=False,
    tempfile=False,
    energyfile=False,
    pressfile=False,
    rdffile=False,
    n_save=10,
    simulation_name="Equilibration part b",
)

final_config = execute.run_simulation(
    initial_config=equilibrated_config,
    integrator="VelocityVerlet",
    force="LJ",
    steps=settings.nsteps_production,
    thermostat="andersen_thermostat",
    thermostat_params=[settings.Tdesired, settings.nu, settings.deltat],
    n_thermostat=1,
    trajfile=False,
    tempfile="b_temp",
    energyfile=False,
    pressfile="b_press",
    rdffile="b_gr",
    n_save=10,
    simulation_name="Production part b",
)

b_temp = np.loadtxt(os.path.join(settings.path, "b_temp.txt"))[:, 1]
b_press = np.loadtxt(os.path.join(settings.path, "b_press.txt"))[:, 1]
b_r, b_gr = np.loadtxt(os.path.join(settings.path, "b_gr.txt")).T
t = np.arange(len(b_temp)) * 2

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.set_xlabel(r"$t$ [fs]")
ax1.set_ylabel(r"$T$ [K]", color="b")
ax1.plot(t, b_temp, color="b", label="Temperature")
ax1.tick_params(axis="y", labelcolor="b")
ax2 = ax1.twinx()
ax2.set_ylabel(
    r"$P$ [$\mathrm{g/mole}\cdot\mathrm{nm}^{-1}\cdot\mathrm{fs}^{-2}$]", color="r"
)
ax2.plot(t, b_press, color="r", label="Pressure")
ax2.tick_params(axis="y", labelcolor="r")
fig.tight_layout()
plt.savefig(os.path.join(settings.path, "b_PT.png"), bbox_inches="tight", dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(b_r, b_gr, color="b")
plt.xlabel(r"$r / \sigma$")
plt.ylabel(r"$g(r)$")
plt.grid()
plt.savefig(os.path.join(settings.path, "b_g.png"), bbox_inches="tight", dpi=300)
plt.show()

num_blocks = 5
block_size = len(b_press) // num_blocks
block_means = np.zeros(num_blocks)
for i in range(num_blocks):
    start = i * block_size
    if i < num_blocks - 1:
        end = (i + 1) * block_size
    else:
        end = len(b_press)
    block_means[i] = np.mean(b_press[start:end])
print(f"b: iii: mean values for each block: {block_means}")
P_mean = np.mean(block_means)
P_error = np.std(block_means, ddof=1) / np.sqrt(num_blocks)
print(f"b: iv: P_mean: {P_mean:.4f}")
print(f"b: iv: P_error: {P_error:.4f}")
rho = settings.rho / sigma**3
B2_sys = B2(beta, settings.eps, sigma)
P_B2 = 1 / beta * rho + 1 / beta * B2_sys * rho**2
print(f"b: v: P_B2: {P_B2:.4f}")
