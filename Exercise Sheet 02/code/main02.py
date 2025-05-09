from simulation import run_simulation
from data_io import export_data

# Initialization
box_bounds = ((0, 10), (0, 10), (0, 10))
initial_configuration = ...

# Equilibration
equilibration_configuration = run_simulation(
    initial_configuration,
    potential="lj_cut",
    potential_params=[0.297741315, 0.188, 2.5],
    dt_max=10000,
    dt=1,
    integrator="velocity_verlet",
    boundary_conditions="perodic",
    thermostat="none",
    T=300,
    dt_thermostat=10,
)

# Production run
equilibrated_configuration = equilibration_configuration[-1]
data = run_simulation(
    equilibrated_configuration,
    potential="lj_cut",
    potential_params=[0.297741315, 0.188, 2.5],
    dt_max=20000,
    dt=1,
    integrator="velocity_verlet",
    boundary_conditions="perodic",
    thermostat="none",
)

export_data(
    data,
    selected_properties=["pID", "type", "x", "y", "z", "vx", "vy", "vz"],
    dt_export=1,
    filename="output.dat",
    box_bounds=box_bounds,
)
