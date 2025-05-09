import numpy as np
import params
from tqdm import tqdm
from integrators import velocity_verlet
from typing import Union

# TODO: Write integrators euler and verlet, thermostat and reboxing functions
# for save
# filename: str = params.filename,
# path: str = params.path,
# selected_properties: list = True,
# dt_export: int = params.dt_export,


def md_simulation(
    initial_configuration: np.ndarray,
    properties: dict = params.properties,
    potential: str = params.potential,
    potential_params: Union[tuple, float] = params.potential_params,
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
    Originally taken from params:
        properties (dict): The featured properties of the particles.
        potential (str): The potential model to use. Options: "lj", "gravitational".
        potential_params (Union[tuple, float]): The parameters for the potential model; lj: epsilon and sigma (tuple), gravitational: G (float).
        integrator (str): The integrator to use. Options: "velocity_verlet", "verlet", "euler".
        dt (int): The time step for the simulation.
        dt_max (int): The maximum time step for the simulation.
        box_bounds (tuple): The bounds of the simulation box.
        boundary_conditions (str): The boundary conditions to use. Options: "periodic", "reflective", "none".
        thermostat (str): The thermostat to use. Options: "none", "berendsen", "nose-hoover".
        T (float): The temperature of the system.

    Returns:
        numpy.ndarray: The data of all configurations in the format (timestep, property, particle) (i.e. shape = (dt_max, len(properties), nparticles)).
    """
    # initialize the data array to store the simulation results
    nparticles = initial_configuration.shape[1]
    data = np.zeros((dt_max, len(properties), nparticles), dtype=object)
    data[0] = initial_configuration.astype(object)
    # choose the integrator function based on the input parameter
    try:
        integrator_func = globals()[f"{integrator}"]
    except KeyError:
        raise ValueError(f"Unknown integrator: {integrator}")
    # update positions and velocities based on the chosen integrator for each time step; written in a seemingly strange way to avoid too many if statements
    print(f"Starting simulation with {integrator} integrator for {dt_max} steps.")
    if boundary_conditions == "none" and thermostat == "none":
        for t in tqdm(range(1, dt_max)):
            data[t] = integrator_func(data, t, dt, potential, potential_params)
        # TODO: Implement arguments, reboxing and thermostat functions here
    elif boundary_conditions != "none" and thermostat == "none":
        ...
    elif boundary_conditions == "none" and thermostat != "none":
        ...
    else:
        ...

    print("Simulation completed.")
    return data
