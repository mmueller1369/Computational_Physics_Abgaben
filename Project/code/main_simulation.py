import numpy as np
import execute
import initialize
import settings
settings.init()


# ----------------- Part 1 ----------------- #
# perturbations_length = np.array([0.05, 0.1, 0, 0, 0.05, 0.1]) * settings.s0
# perturbations_angle = np.array([0, 0, 0.05, 0.1, 0.05, 0.1]) * settings.theta0
# names = ["pert_s_0.05", "pert_s_0.1", "pert_t_0.05", "pert_t_0.1", "pert_b_0.05", "pert_b_0.1"]
perturbations_length = [0.07 * settings.s0]
perturbations_angle = [0.02 * settings.theta0]
names = ["long_pert_s_0.1"]

for pert_length, pert_angle, name in zip(perturbations_length, perturbations_angle, names):
    x, y, z, vx, vy, vz = initialize.single(pert_length=pert_length, pert_angle=pert_angle)
    f_initial = np.zeros(shape=(3))
    initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]
    settings.bounds = np.array([[-1,1], [-1,1], [-1,1]])

    execute.run_simulation(
        initial_config=initial_config,
        force="H2O",
        force_params=[settings.k_bond, settings.s0, # Intramolecular parameters
            settings.k_angle, settings.theta0,
            settings.eps, settings.sigma, settings.cutoff, # Intramolecular parameters
            settings.qO, settings.qH, settings.eps0_el, settings.alpha],
        steps=40000,
        thermostat=False,
        trajfile=f"part_1/{name}_traj",
        energyfile=f"part_1/{name}_energy",
        n_save=1,
        simulation_name=f"Part 1 - {name}",
    )


# ----------------- Part 2 ----------------- #
# x, y, z, vx, vy, vz = initialize.cubic_lattice()
# f_initial = np.zeros(shape=(3))
# initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]

# settings.deltat /= 10
# # settings.eps0_el /= 10
# execute.run_simulation(
#     initial_config=initial_config,
#     force="H2O",
#     force_params=[settings.k_bond, settings.s0, # Intramolecular parameters
#         settings.k_angle, settings.theta0,
#         settings.eps, settings.sigma, settings.cutoff, # Intramolecular parameters
#         settings.qO, settings.qH, settings.eps0_el, settings.alpha],
#     steps=100000,
#     thermostat="Berendsen",
#     thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
#     trajfile=f"part_2/test_traj",
#     tempfile=f"part_2/test_temp",
#     energyfile=f"part_2/test_energy",
#     n_save=10,
#     simulation_name=f"Part 2 - test",
# )