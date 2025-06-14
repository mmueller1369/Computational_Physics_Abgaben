import execute
import settings
import initialize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

settings.init()


# ----------------- Part a ----------------- #
# def B2(beta, eps, sigma):
#     def potentialLJ(r, eps, sigma):
#         mask = r > 0
#         sf6 = np.zeros_like(r)
#         sf6[mask] = (sigma / r[mask]) ** 6
#         epot = np.zeros_like(r)
#         epot[mask] = 4.0 * eps * sf6[mask] * (sf6[mask] - 1.0)
#         return epot

#     cutoff = (settings.xhi - settings.xlo) * 3
#     deltar = cutoff / 10000
#     r = np.arange(0, cutoff, deltar)
#     U = potentialLJ(r, eps, sigma)
#     integral_core = np.exp(-beta * U) - 1
#     integral_core *= 4 * np.pi * r**2  # conversion into spherical coordinates
#     integral = -1 / 2 * np.sum(integral_core) * deltar
#     return integral


# beta = 1 / (settings.Tdesired * settings.kb)
# sigma = settings.sigma
# print(
#     f"a: for our parameter set (argon): B2/eps = {B2(
#         beta,
#         1,  # set to unity
#         sigma,
#     ):.4f}"
# )

# epss = np.linspace(0, 1, 1000) / beta
# B2_eps = [B2(beta, eps, sigma) / sigma**3 for eps in epss]

# plt.figure(figsize=(6, 4))
# plt.plot(epss * beta, B2_eps, label=r"$B_2/\sigma^3$", color="b")
# plt.axvline(settings.eps * beta, label=r"$\epsilon\beta_\mathrm{Argon}$", color="r")
# plt.xlabel(r"$\epsilon\beta$")
# plt.ylabel(r"$B_2/\sigma^3$")
# plt.grid()
# plt.legend()
# plt.savefig(os.path.join(settings.path, "a.png"), bbox_inches="tight", dpi=300)
# plt.show()


# ----------------- Part b ----------------- #
# x, y, z, vx, vy, vz = initialize.InitializeAtoms()
# f_initial = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
# initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]
# equilibrated_config = execute.run_simulation(
#     initial_config=initial_config,
#     integrator="VelocityVerlet",
#     force="LJ",
#     steps=settings.nsteps_equi,
#     thermostat="andersen_thermostat",
#     thermostat_params=[settings.Tdesired, settings.nu, settings.deltat],
#     n_thermostat=1,
#     trajfile=False,
#     tempfile=False,
#     energyfile=False,
#     pressfile=False,
#     rdffile=False,
#     n_save=10,
#     simulation_name="Equilibration part b",
# )

# final_config = execute.run_simulation(
#     initial_config=equilibrated_config,
#     integrator="VelocityVerlet",
#     force="LJ",
#     steps=settings.nsteps_production,
#     thermostat="andersen_thermostat",
#     thermostat_params=[settings.Tdesired, settings.nu, settings.deltat],
#     n_thermostat=1,
#     trajfile=False,
#     tempfile="b_temp",
#     energyfile=False,
#     pressfile="b_press",
#     rdffile="b_gr",
#     n_save=10,
#     simulation_name="Production part b",
# )

# b_temp = np.loadtxt(os.path.join(settings.path, "b_temp.txt"))[:, 1]
# b_press = np.loadtxt(os.path.join(settings.path, "b_press.txt"))[:, 1]
# b_r, b_gr = np.loadtxt(os.path.join(settings.path, "b_gr.txt")).T
# t = np.arange(len(b_temp)) * 2

# fig, ax1 = plt.subplots(figsize=(7, 4))
# ax1.set_xlabel(r"$t$ [fs]")
# ax1.set_ylabel(r"$T$ [K]", color="b")
# ax1.plot(t, b_temp, color="b", label="Temperature")
# ax1.tick_params(axis="y", labelcolor="b")
# ax2 = ax1.twinx()
# ax2.set_ylabel(
#     r"$P$ [$\mathrm{g/mole}\cdot\mathrm{nm}^{-1}\cdot\mathrm{fs}^{-2}$]", color="r"
# )
# ax2.plot(t, b_press, color="r", label="Pressure")
# ax2.tick_params(axis="y", labelcolor="r")
# fig.tight_layout()
# plt.savefig(os.path.join(settings.path, "b_PT.png"), bbox_inches="tight", dpi=300)
# plt.show()

# plt.figure(figsize=(7, 5))
# plt.plot(b_r, b_gr, color="b")
# plt.xlabel(r"$r / \sigma$")
# plt.ylabel(r"$g(r)$")
# plt.grid()
# plt.savefig(os.path.join(settings.path, "b_g.png"), bbox_inches="tight", dpi=300)
# plt.show()


def blog_averages(data, num_blocks):
    num_blocks = 5
    block_size = len(data) // num_blocks
    block_means = np.zeros(num_blocks)
    for i in range(num_blocks):
        start = i * block_size
        if i < num_blocks - 1:
            end = (i + 1) * block_size
        else:
            end = len(data)
        block_means[i] = np.mean(data[start:end])
    mean = np.mean(block_means)
    error = np.std(block_means, ddof=1) / np.sqrt(num_blocks)
    return block_means, mean, error


# block_means, P_mean, P_error = blog_averages(b_press, 5)
# print(f"b: iii: mean values for each block: {block_means}")
# print(f"b: iv: P_mean: {P_mean:.4f}")
# print(f"b: iv: P_error: {P_error:.4f}")
# rho = settings.rho / sigma**3
# B2_sys = B2(beta, settings.eps, sigma)
# P_B2 = 1 / beta * rho + 1 / beta * B2_sys * rho**2
# print(f"b: v: P_B2: {P_B2:.4f}")


# ----------------- Part c ----------------- #
rhos = np.logspace(np.log10(0.05), np.log10(0.25), 2)
colors = mpl.cm.viridis(np.linspace(0, 1, len(rhos)))
beta = 1 / (settings.Tdesired * settings.kb)
for r in rhos:
    settings.rho = r
    hi = settings.n1 / (r ** (1 / 3))
    settings.xhi = hi * settings.sigma
    settings.yhi = hi * settings.sigma
    settings.zhi = hi * settings.sigma
    settings.deltaxyz = settings.xhi / settings.n1
    settings.rmax = 1 / 2 * settings.xhi
    x, y, z, vx, vy, vz = initialize.InitializeAtoms()
    f_initial = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
    initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]
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
        simulation_name=f"Equilibration part c, rho = {r:.3f}",
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
        tempfile=False,
        energyfile=False,
        pressfile=f"c_press_rho={r:.3f}",
        rdffile=f"c_gr_rho={r:.3f}",
        n_save=10,
        simulation_name=f"Production part c, rho = {r:.3f}",
    )


P_mean, P_error = np.zeros(len(rhos)), np.zeros(len(rhos))
for i, r in enumerate(rhos):
    press = np.loadtxt(os.path.join(settings.path, f"c_press_rho={r:.3f}.txt"))[:, 1]
    block_means, mean, error = blog_averages(press, 5)
    P_mean[i] = mean
    P_error[i] = error
# P_mean -= rhos/beta

plt.figure(figsize=(7, 5))
plt.errorbar(rhos, P_mean, P_error, capsize=3)
plt.xlabel(r"$\rho$")
plt.ylabel(r"$P$ [$\mathrm{g/mole}\cdot\mathrm{nm}^{-1}\cdot\mathrm{fs}^{-2}$]")
plt.grid()
plt.loglog()
plt.savefig(os.path.join(settings.path, "c_P.png"), bbox_inches="tight", dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
for r, color in zip(rhos, colors):
    dist = np.loadtxt(os.path.join(settings.path, f"c_gr_rho={r:.3f}.txt"))[:, 0]
    g = np.loadtxt(os.path.join(settings.path, f"c_gr_rho={r:.3f}.txt"))[:, 1]
    plt.plot(dist, g, color=color, label=rf"$\rho={r}$")
plt.xlabel(r"$r / \sigma$")
plt.ylabel(r"$g(r)$")
plt.grid()
plt.legend()
plt.savefig(os.path.join(settings.path, "c_g.png"), bbox_inches="tight", dpi=300)
plt.show()
