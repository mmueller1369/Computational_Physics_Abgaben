from simulation import run_simulation
from data_io import export_data
from part_a import (
    lj_system_initial_configuration,
    create_geometry,
    assign_random_velocities,
)
import params


## Initialization
box_bounds = params.box_bounds
initial_configuration = lj_system_initial_configuration
initial_configuration = create_geometry(initial_configuration)
initial_configuration = assign_random_velocities(initial_configuration)
## Equilibration
params.dt_max = 5000
equilibration_configuration = run_simulation(
    initial_configuration,
    potential=params.potential,
    potential_params=params.potential_params,
    dt_max=params.dt_max,
    dt=params.dt,
    integrator=params.integrator,
    boundary_conditions=params.boundary_conditions,
    thermostat="velocity_rescaling",
    T=params.T,
    dt_thermostat=params.dt_thermostat,
)

export_data(
    equilibration_configuration,
    selected_properties=["pID", "type", "x", "y", "z", "vx", "vy", "vz", "mass"],
    dt_export=1,
    filename="equilibration.dat",
    box_bounds=box_bounds,
)

## Production run
equilibrated_configuration = equilibration_configuration[-1]
params.dt_max = 20000
data = run_simulation(
    equilibrated_configuration,
    potential=params.potential,
    potential_params=params.potential_params,
    dt_max=params.dt_max,
    dt=params.dt,
    integrator=params.integrator,
    boundary_conditions=params.boundary_conditions,
    thermostat="none",
)

## Export data
export_data(
    data,
    selected_properties=["pID", "type", "x", "y", "z", "vx", "vy", "vz", "mass"],
    dt_export=1,
    filename="debug.dat",
    box_bounds=box_bounds,
)
