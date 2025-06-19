import execute
import settings
import initialize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import tools_6

settings.init()

# ----------------- Part b ----------------- #
x, y, z, vx, vy, vz = initialize.InitializeAtomsBond()
f_initial = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3 *2))
initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]
equilibrated_config = execute.run_simulation_bonded(
    initial_config=initial_config,
    integrator="VelocityVerletBond",
    force="LJBond",
    steps=settings.nsteps_equi,
    thermostat="berendsen_thermostat",
    thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
    n_thermostat=1,
    trajfile1=False,
    trajfile2=False,
    energyfile=False,
    rdffile1=False,
    rdffile2=False,
    n_save1=1,
    n_save2=50,
    simulation_name="Equilibration",
)
final_config = execute.run_simulation_bonded(
    initial_config=equilibrated_config,
    integrator="VelocityVerletBond",
    force="LJBond",
    steps=settings.nsteps_production,
    thermostat="berendsen_thermostat",
    thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
    n_thermostat=1,
    trajfile1='TrajectoryforC',
    trajfile2='TrajectoryforD',
    energyfile='Energy',
    rdffile1='g(r)_1',
    rdffile2='g(r)_2',
    n_save1=1,
    n_save2=50,
    simulation_name="Production",
)

# b_temp = np.loadtxt(os.path.join(settings.path, "b_temp.txt"))[:, 1]
# b_press = np.loadtxt(os.path.join(settings.path, "b_press.txt"))[:, 1]
# b_r, b_gr = np.loadtxt(os.path.join(settings.path, "b_gr.txt")).T
# t = np.arange(len(b_temp)) * 2
# sigma = settings.sigma

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


# block_means, P_mean, P_error = tools_6.blog_averages(b_press, 5)
# beta = 1 / (settings.Tdesired * settings.kb)
# print(f"b: iii: mean values for each block: {block_means}")
# print(f"b: iv: P_mean: {P_mean:.4f}")
# print(f"b: iv: P_error: {P_error:.4f}")
# rho = settings.rho / sigma**3
# B2_sys = tools_6.B2_red(beta, settings.eps) * sigma**3
# P_B2 = 1 / beta * rho + 1 / beta * B2_sys * rho**2
# print(f"b: v: P_B2: {P_B2:.4f}")


# # ----------------- Part c ----------------- #
# rhos = np.logspace(np.log10(0.005), np.log10(0.25), 7)
# for r in rhos:
#     settings.rho = r
#     settings.lx = settings.n1 / (settings.rho ** (1 / 3))
#     settings.ly = settings.n2 / (settings.rho ** (1 / 3))
#     settings.lz = settings.n3 / (settings.rho ** (1 / 3))
#     settings.nparticles = settings.n1 * settings.n2 * settings.n3
#     settings.xhi = settings.lx * settings.sigma
#     settings.yhi = settings.ly * settings.sigma
#     settings.zhi = settings.lz * settings.sigma
#     settings.deltaxyz = settings.xhi / settings.n1
#     settings.rmax = 1 / 2 * settings.xhi

#     x, y, z, vx, vy, vz = initialize.InitializeAtoms()
#     f_initial = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
#     initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]
#     equilibrated_config = execute.run_simulation(
#         initial_config=initial_config,
#         integrator="VelocityVerlet",
#         force="LJ",
#         steps=settings.nsteps_equi,
#         thermostat="andersen_thermostat",
#         thermostat_params=[settings.Tdesired, settings.nu, settings.deltat],
#         n_thermostat=1,
#         trajfile=False,
#         tempfile=False,
#         energyfile=False,
#         pressfile=False,
#         rdffile=False,
#         n_save=10,
#         simulation_name=f"Equilibration part c, rho = {r:.3f}",
#     )

#     final_config = execute.run_simulation(
#         initial_config=equilibrated_config,
#         integrator="VelocityVerlet",
#         force="LJ",
#         steps=settings.nsteps_production,
#         thermostat="andersen_thermostat",
#         thermostat_params=[settings.Tdesired, settings.nu, settings.deltat],
#         n_thermostat=1,
#         trajfile=False,
#         tempfile=False,
#         energyfile=False,
#         pressfile=f"c_press_rho={r:.3f}",
#         rdffile=f"c_gr_rho={r:.3f}",
#         n_save=10,
#         simulation_name=f"Production part c, rho = {r:.3f}",
#     )


# P_mean, P_error = np.zeros(len(rhos)), np.zeros(len(rhos))
# for i, r in enumerate(rhos):
#     press = np.loadtxt(os.path.join(settings.path, f"c_press_rho={r:.3f}.txt"))[:, 1]
#     block_means, mean, error = tools_6.blog_averages(press, 5)
#     P_mean[i] = mean
#     P_error[i] = error
# beta = 1 / (settings.Tdesired * settings.kb)
# P_id = rhos / beta / settings.sigma**3

# plt.figure(figsize=(7, 5))
# plt.errorbar(rhos, P_mean, P_error, capsize=3, label=r"$\langle P\rangle$")
# plt.plot(rhos, P_id, label=r"$P_\mathrm{id}$")
# plt.xlabel(r"$\rho$")
# plt.ylabel(r"$P$ [$\mathrm{g/mole}\cdot\mathrm{nm}^{-1}\cdot\mathrm{fs}^{-2}$]")
# plt.grid()
# plt.loglog()
# plt.legend()
# plt.savefig(os.path.join(settings.path, "c_P_pure.png"), bbox_inches="tight", dpi=300)
# plt.show()


# plt.figure(figsize=(7, 5))
# plt.errorbar(rhos, P_mean - P_id, P_error, capsize=3, label=r"$\langle P\rangle$")
# plt.xlabel(r"$\rho$")
# plt.ylabel(r"$P$ [$\mathrm{g/mole}\cdot\mathrm{nm}^{-1}\cdot\mathrm{fs}^{-2}$]")
# plt.grid()
# plt.loglog()
# plt.savefig(os.path.join(settings.path, "c_P_sub.png"), bbox_inches="tight", dpi=300)
# plt.show()

# colors = mpl.cm.viridis(np.linspace(0, 1, len(rhos)))
# plt.figure(figsize=(7, 5))
# for r, color in zip(rhos, colors):
#     dist = np.loadtxt(os.path.join(settings.path, f"c_gr_rho={r:.3f}.txt"))[:, 0]
#     g = np.loadtxt(os.path.join(settings.path, f"c_gr_rho={r:.3f}.txt"))[:, 1]
#     plt.plot(dist, g, color=color, label=rf"$\rho={r:.3f}$")
# plt.xlabel(r"$r / \sigma$")
# plt.ylabel(r"$g(r)$")
# plt.xlim(0, 10)
# plt.grid()
# plt.legend()
# plt.savefig(os.path.join(settings.path, "c_g.png"), bbox_inches="tight", dpi=300)
# plt.show()
