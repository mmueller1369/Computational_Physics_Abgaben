from simulation_tools import md_simulation
from data_io import export_data
import solar_system

initial_configuration = solar_system.solar_system_initial_configuration

data = md_simulation(
    initial_configuration,
    potential="gravitational",
    potential_params=1.488136e-5,
    dt_max=1000,
    dt=0.5,
    integrator="velocity_verlet",
    boundary_conditions="none",
)

export_data(
    data,
    selected_properties=["pID", "type", "x", "y", "z", "vx", "vy", "vz"],
    dt_export=10,
    filename="output.dat",
    box_bounds=((-10, 10), (-10, 10), (-10, 10)),
)
