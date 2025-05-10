import numpy as np
from simulation import run_simulation
from data_io import export_data
from part_a import (
    lj_system_initial_configuration,
    create_geometry,
    assign_random_velocities,
)
import params

# Simulations needed for exercise e:
params.dt_thermostat = 10
params.thermostat = "velocity_rescaling"
params.dt_export = 1

## Initialization
initial_configuration = lj_system_initial_configuration
initial_configuration = create_geometry(initial_configuration)
initial_configuration = assign_random_velocities(initial_configuration)

for dt_scale in [1, 10, 0.1]:
    ## Equilibration
    params.dt_max = 10000
    dt = params.dt * dt_scale

    equilibration_configuration = run_simulation(
        initial_configuration,
        potential_params=params.potential_params,
        dt=dt,
        dt_max=params.dt_max,
        thermostat=params.thermostat,
        dt_thermostat=params.dt_thermostat,
    )
    export_data(
        equilibration_configuration,
        selected_properties=["pID", "type", "x", "y", "z", "vx", "vy", "vz", "mass"],
        filename=f"equi_{dt_scale}.dat",
    )

    ## Production run
    equilibrated_configuration = equilibration_configuration[-1]
    params.dt_max = 20000

    data = run_simulation(
        equilibrated_configuration,
        potential_params=params.potential_params,
        dt=dt,
        dt_max=params.dt_max,
        thermostat="none",
    )

    ## Export data
    export_data(
        data,
        selected_properties=["pID", "type", "x", "y", "z", "vx", "vy", "vz", "mass"],
        filename=f"exercise_e_dt={dt_scale}.dat",
        dt_export=params.dt_export,
    )


# Simulation needed for exercise h:
params.dt_max = 10000
for cutoff in [3.25, 4.0]:
    params.potential_params = [params.epsilon, params.sigma, cutoff]

    ## Equilibration
    equilibration_configuration = run_simulation(
        initial_configuration,
        potential_params=params.potential_params,
        dt=params.dt,
        dt_max=params.dt_max,
        thermostat=params.thermostat,
        dt_thermostat=params.dt_thermostat,
    )

    ## Production run
    equilibrated_configuration = equilibration_configuration[-1]
    data = run_simulation(
        equilibrated_configuration,
        potential_params=params.potential_params,
        dt=params.dt,
        dt_max=params.dt_max,
        thermostat="none",
    )

    ## Export data
    export_data(
        data,
        selected_properties=["pID", "type", "x", "y", "z", "vx", "vy", "vz", "mass"],
        filename=f"exercise_h_cut={cutoff}.dat",
        dt_export=params.dt_export,
    )


# Simulation needed for exercise i:


for multiplier in [0.1, 0.5, 1, 5]:
    epsilon = params.kB * params.T * multiplier
    params.potential_params = [epsilon, params.sigma, 2.5]

    ## Equilibration
    equilibration_configuration = run_simulation(
        initial_configuration,
        potential_params=params.potential_params,
        dt=params.dt,
        dt_max=params.dt_max,
        thermostat=params.thermostat,
        dt_thermostat=params.dt_thermostat,
    )

    ## Production run
    equilibrated_configuration = equilibration_configuration[-1]
    data = run_simulation(
        initial_configuration,
        potential_params=params.potential_params,
        dt=params.dt,
        dt_max=params.dt_max,
        thermostat="none",
    )

    ## Export data
    export_data(
        data,
        selected_properties=["pID", "type", "x", "y", "z", "vx", "vy", "vz", "mass"],
        filename=f"exercise_i_multiplier={multiplier}.dat",
    )

# fg
