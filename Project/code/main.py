import numpy as np
import execute
import initialize
import settings
settings.init()


# ----------------- Part 1 ----------------- #
x, y, z, vx, vy, vz = initialize.single()
f_initial = np.zeros(shape=(3))
initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]
settings.bounds = np.array([[-1,1], [-1,1], [-1,1]])
print(initial_config)

execute.run_simulation(
    initial_config=initial_config,
    force="H2O",
    force_params=[settings.k_bond, settings.s0, # Intramolecular parameters
        settings.k_angle, settings.theta0,
        settings.eps, settings.sigma, settings.cutoff, # Intramolecular parameters
        settings.qO, settings.qH, settings.eps0_el, settings.alpha],
    steps=1000,
    thermostat="BerendsenThermostat",
    thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
    trajfile="part1_traj",
    n_save=1,
    simulation_name="Part 1",
)