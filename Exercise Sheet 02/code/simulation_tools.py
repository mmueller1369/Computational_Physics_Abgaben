import numpy as np
import params
import tqdm

# TODO: Write integrator, potential, thermostat and reboxing functions
# filename: str = params.filename,
# path: str = params.path,
# selected_properties: list = True,
# dt_export: int = params.dt_export,


def md_simulation(
    initial_configuration: np.ndarray,
    properties: dict = params.properties,
    potential: str = params.potential,
    integrator: str = params.integrator,
    dt: int = params.dt,
    dt_max: int = params.dt_max,
    box_bounds: tuple = params.box_bounds,
    boundary_conditions: str = params.boundary_conditions,
    thermostat: str = params.thermostat,
    T: float = params.T,
) -> np.ndarray:
    """
    Run a molecular dynamics simulation.

    Parameters:
        initial_configuration (numpy.ndarray): The initial configuration of the system featuring the defined properties; format: (properties, particles).
        properties (dict): The featured properties of the particles. Originally taken from params.
        potential (str): The potential model to use. Originally taken from params. Options: "lj", "gravitational".
        integrator (str): The integrator to use. Originally taken from params. Options: "velocity_verlet", "verlet", "euler".
        dt (int): The time step for the simulation. Originally taken from params.
        dt_max (int): The maximum time step for the simulation. Originally taken from params.
        box_bounds (tuple): The bounds of the simulation box. Originally taken from params.
        boundary_conditions (str): The boundary conditions to use. Originally taken from params. Options: "periodic", "reflective", "open".
        thermostat (str): The thermostat to use. Originally taken from params. Options: "none", "berendsen", "nose-hoover".
        T (float): The temperature of the system. Originally taken from params.

    Returns:
        numpy.ndarray: The data of all configurations in the format (timestep, property, particle) (i.e. shape = (dt_max, len(properties), nparticles)).
    """
    # initialize the data array to store the simulation results
    nparticles = initial_configuration.shape[1]
    data = np.zeros((dt_max, len(properties), nparticles))
    data[0] = initial_configuration
    # choose the integrator function based on the input parameter
    if integrator == "velocity_verlet":
        integrator_func = velocity_verlet
    # update positions and velocities based on the chosen integrator for each time step
    print(f"Starting simulation with {integrator} integrator for {dt_max} steps.")
    for t in tqdm(range(1, dt_max)):
        data[t] = integrator_func(data[t - 1], ...)
        # TODO: Implement reboxing and thermostat functions here
    print("Simulation completed.")
    return data
