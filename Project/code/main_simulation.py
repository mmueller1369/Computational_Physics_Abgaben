import numpy as np
import execute
import initialize
import settings
# import settings_SI as settings
settings.init()


# ----------------- Part 1 ----------------- #
# perturbations compared to the equilibrium
perturbations_length = np.array([0.05, 0.1, 0, 0, 0.05, 0.1]) * settings.s0
perturbations_angle = np.array([0, 0, 0.05, 0.1, 0.05, 0.1]) * settings.theta0
# naming: s: only length perturbated; t: only angle; b. both
names = ["pert_s_0.05", "pert_s_0.1", "pert_t_0.05", "pert_t_0.1", "pert_b_0.05", "pert_b_0.1"]

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


# ----------------- Part 2 ----------------- #
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
    steps=100000,
    thermostat="Berendsen",
    thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
    trajfile=f"part_3/traj_eq",
    energyfile=f"part_3/energy_eq",
    tempfile=f"part_3/temp_eq",
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
    steps=100000,
    # thermostat="Berendsen",
    # thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
    thermostat=False,
    trajfile=f"part_3/traj",
    energyfile=f"part_3/energy",
    tempfile=f"part_3/temp",
    n_save=10,
    simulation_name=f"Part 3 - run",
)