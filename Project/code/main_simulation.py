import numpy as np
import execute
import initialize
import settings
settings.init()


# # ----------------- Part 1 ----------------- #
# # perturbations compared to the equilibrium
# perturbations_length = np.array([0.05, 0.1, 0, 0, 0.05, 0.1]) * settings.s0
# perturbations_angle = np.array([0, 0, 0.05, 0.1, 0.05, 0.1]) * settings.theta0
# # naming: s: only length perturbated; t: only angle; b. both
# names = ["pert_s_0.05", "pert_s_0.1", "pert_t_0.05", "pert_t_0.1", "pert_b_0.05", "pert_b_0.1"]

# names = ["pert_b_0.1"]
# perturbations_length = [.01]
# perturbations_angle = [10.45 * np.pi/180]

# for pert_length, pert_angle, name in zip(perturbations_length, perturbations_angle, names):
#     x, y, z, vx, vy, vz = initialize.single(pert_length=pert_length, pert_angle=pert_angle)
#     f_initial = np.zeros(shape=(3))
#     initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]
#     settings.bounds = np.array([[-1,1], [-1,1], [-1,1]])

#     execute.run_simulation(
#         initial_config=initial_config,
#         force="H2O",
#         force_params=[settings.k_bond, settings.s0, # Intramolecular parameters
#             settings.k_angle, settings.theta0,
#             settings.eps, settings.sigma, settings.cutoff, # Intramolecular parameters
#             settings.qO, settings.qH, settings.eps0_el, settings.alpha],
#         steps=10000,
#         thermostat=False,
#         trajfile=f"part_1/{name}_traj",
#         energyfile=f"part_1/{name}_energy",
#         n_save=1,
#         simulation_name=f"Part 1 - {name}",
#     )



# ----------------- Part 3 ----------------- #
folder = "part_3"
folder = "prange"
x, y, z, vx, vy, vz = initialize.cubic_lattice()
f_initial = np.zeros(shape=(3))
initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]

equilibrated_config = execute.run_simulation(
    initial_config=initial_config,
    force="H2O",
    force_params=[settings.k_bond, settings.s0, # Intramolecular parameters
        settings.k_angle, settings.theta0,
        settings.eps, settings.sigma, settings.cutoff, # Intramolecular parameters
        settings.qO, settings.qH, settings.eps0_el, settings.alpha],
    steps=15000,
    thermostat="Berendsen",
    thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
    trajfile=f"{folder}/traj_eq",
    energyfile=f"{folder}/energy_eq",
    tempfile=f"{folder}/temp_eq",
    n_save=10,
    simulation_name=f"Part 3 - eq",
)

execute.run_simulation(
    initial_config=equilibrated_config,
    force="H2O",
    force_params=[settings.k_bond, settings.s0, # Intramolecular parameters
        settings.k_angle, settings.theta0,
        settings.eps, settings.sigma, settings.cutoff, # Intramolecular parameters
        settings.qO, settings.qH, settings.eps0_el, settings.alpha],
    steps=40000,
    # thermostat="Berendsen",
    # thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
    thermostat=False,
    trajfile=f"{folder}/traj",
    energyfile=f"{folder}/energy",
    tempfile=f"{folder}/temp",
    n_save=10,
    simulation_name=f"Part 3 - run",
)



# # ----------------- Part 4 ----------------- #
# folder = "part_4"
# x, y, z, vx, vy, vz = initialize.cubic_lattice()
# f_initial = np.zeros(shape=(3))
# initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]

# equilibrated_config = execute.run_simulation(
#     initial_config=initial_config,
#     force="H2O",
#     force_params=[settings.k_bond, settings.s0, # Intramolecular parameters
#         settings.k_angle, settings.theta0,
#         settings.eps, settings.sigma, settings.cutoff, # Intramolecular parameters
#         settings.qO, settings.qH, settings.eps0_el, settings.alpha],
#     steps=15000,
#     thermostat="Berendsen",
#     thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
#     trajfile=f"{folder}/traj_eq",
#     energyfile=f"{folder}/energy_eq",
#     tempfile=f"{folder}/temp_eq",
#     n_save=10,
#     simulation_name=f"Part 4 - eq",
# )

# execute.run_simulation(
#     initial_config=equilibrated_config,
#     force="H2O",
#     force_params=[settings.k_bond, settings.s0, # Intramolecular parameters
#         settings.k_angle, settings.theta0,
#         settings.eps, settings.sigma, settings.cutoff, # Intramolecular parameters
#         settings.qO, settings.qH, settings.eps0_el, settings.alpha],
#     steps=2000000,
#     # thermostat="Berendsen",
#     # thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
#     thermostat=False,
#     trajfile=f"{folder}/traj",
#     energyfile=f"{folder}/energy",
#     tempfile=f"{folder}/temp",
#     n_save=1000,
#     simulation_name=f"Part 4 - run",
# )


# # ----------------- Part 5 ----------------- #
# folder = "part_5"
# mixing_rule = "Lorentz-Berthelot"  # or "Geometric"
# x, y, z, vx, vy, vz = initialize.cubic_lattice_salt()
# f_initial = np.zeros(shape=(3))
# initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]

# equilibrated_config = execute.run_simulation(
#     initial_config=initial_config,
#     force="Salt",
#     force_params=[settings.k_bond, settings.s0, # Intramolecular parameters
#         settings.k_angle, settings.theta0,
#         settings.eps, settings.sigma, settings.cutoff, # Intramolecular parameters
#         settings.qO, settings.qH, settings.eps0_el, settings.alpha,
#         settings.eps_Na, settings.sigma_Na, settings.eps_I, settings.sigma_I, # Salt parameters
#         settings.cutoff_salt, settings.qNa, settings.qI, settings.alpha_salt,
#         mixing_rule],
#     steps=30000,
#     masses=settings.masses_salt,
#     thermostat="Berendsen",
#     thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
#     trajfile=f"{folder}/{mixing_rule}_traj_eq",
#     energyfile=f"{folder}/{mixing_rule}_energy_eq",
#     tempfile=f"{folder}/{mixing_rule}_temp_eq",
#     n_save=10,
#     simulation_name=f"Part 5 - eq",
# )

# execute.run_simulation(
#     initial_config=equilibrated_config,
#     force="Salt",
#     force_params=[settings.k_bond, settings.s0, # Intramolecular parameters
#         settings.k_angle, settings.theta0,
#         settings.eps, settings.sigma, settings.cutoff, # Intramolecular parameters
#         settings.qO, settings.qH, settings.eps0_el, settings.alpha,
#         settings.eps_Na, settings.sigma_Na, settings.eps_I, settings.sigma_I, # Salt parameters
#         settings.cutoff_salt, settings.qNa, settings.qI, settings.alpha_salt,
#         mixing_rule],
#     steps=40000,
#     masses=settings.masses_salt,
#     # thermostat="Berendsen",
#     # thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
#     thermostat=False,
#     trajfile=f"{folder}/{mixing_rule}_traj",
#     energyfile=f"{folder}/{mixing_rule}_energy",
#     tempfile=f"{folder}/{mixing_rule}_temp",
#     n_save=10,
#     simulation_name=f"Part 5 - run",
# )