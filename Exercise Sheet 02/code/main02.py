from simulation import run_simulation
from data_io import export_data

# Simulations needed for exercise e:

## Definition of the parameters
epsilon = 0.297741315  # kcal/mol
sigma = 0.188  # nm
cutoff = 2.5  # in sigma
mass = 39.95  # g/mol
kB = 0.0019849421  # kcal/mol/K
T = 300  # K
potential = "lj_cut"
potential_params = [epsilon, sigma, cutoff]
integrator = "velocity_verlet"
boundary_conditions = "periodic"

## Initialization
box_bounds = ((0, 10), (0, 10), (0, 10))
initial_configuration = ...

for dt in [1, 10, 0.1]:
    filename = f"exercise_e_dt={dt}.dat"

    ## Equilibration
    equilibration_configuration = run_simulation(
        initial_configuration,
        potential=potential,
        potential_params=potential_params,
        dt_max=10000,
        dt=dt,
        integrator=integrator,
        boundary_conditions=boundary_conditions,
        thermostat="velocity_rescaling",
        T=T,
        dt_thermostat=10,
    )

    ## Production run
    equilibrated_configuration = equilibration_configuration[-1]
    data = run_simulation(
        equilibrated_configuration,
        potential=potential,
        potential_params=potential_params,
        dt_max=20000,
        dt=dt,
        integrator=integrator,
        boundary_conditions=boundary_conditions,
        thermostat="none",
    )

    ## Export data
    export_data(
        data,
        selected_properties=["pID", "type", "x", "y", "z", "vx", "vy", "vz", "mass"],
        dt_export=1,
        filename=filename,
        box_bounds=box_bounds,
    )


# Simulation needed for exercise h:
dt = 1

for cutoff in [3.25, 4.0]:
    potential_params = [epsilon, sigma, cutoff]
    filename = f"exercise_h_cut={cutoff}.dat"

    ## Equilibration
    equilibration_configuration = run_simulation(
        initial_configuration,
        potential=potential,
        potential_params=potential_params,
        dt_max=10000,
        dt=dt,
        integrator=integrator,
        boundary_conditions=boundary_conditions,
        thermostat="velocity_rescaling",
        T=T,
        dt_thermostat=10,
    )

    ## Production run
    equilibrated_configuration = equilibration_configuration[-1]
    data = run_simulation(
        equilibrated_configuration,
        potential=potential,
        potential_params=potential_params,
        dt_max=10000,
        dt=dt,
        integrator=integrator,
        boundary_conditions=boundary_conditions,
        thermostat="none",
    )

    ## Export data
    export_data(
        data,
        selected_properties=["pID", "type", "x", "y", "z", "vx", "vy", "vz", "mass"],
        dt_export=1,
        filename=filename,
        box_bounds=box_bounds,
    )


# Simulation needed for exercise i:
dt = 1
cutoff = 2.5

for multiplier in [0.1, 0.5, 1, 5]:
    epsilon = multiplier * kB * T
    potential_params = [epsilon, sigma, cutoff]
    filename = f"exercise_i_multiplier={multiplier}.dat"

    ## Equilibration
    equilibration_configuration = run_simulation(
        initial_configuration,
        potential=potential,
        potential_params=potential_params,
        dt_max=10000,
        dt=dt,
        integrator=integrator,
        boundary_conditions=boundary_conditions,
        thermostat="velocity_rescaling",
        T=T,
        dt_thermostat=10,
    )

    ## Production run
    equilibrated_configuration = equilibration_configuration[-1]
    data = run_simulation(
        equilibrated_configuration,
        potential=potential,
        potential_params=potential_params,
        dt_max=10000,
        dt=dt,
        integrator=integrator,
        boundary_conditions=boundary_conditions,
        thermostat="none",
    )

    ## Export data
    export_data(
        data,
        selected_properties=["pID", "type", "x", "y", "z", "vx", "vy", "vz", "mass"],
        dt_export=1,
        filename=filename,
        box_bounds=box_bounds,
    )

# fg
