import numpy as np
import params
from forces import force_matrix
from typing import Union


def velocity_verlet(
    data: np.ndarray,
    timestep: int,
    dt: float,
    potential: str,
    potential_params: Union[tuple, float],
    box_bounds: tuple,
    boundary_conditions: str,
) -> np.ndarray:
    """
    Velocity Verlet integrator.

    Parameters:
        data (numpy.ndarray): The overall data array of the simulation; format: (timesteps, properties, particles).
        timestep (int): The current timestep of the simulation.
        dt (float): The time step for the simulation.
        potential (str): The potential model to use. Options: "lj", "gravitational".
        potential_params (list): The parameters for the potential model; lj: epsilon and sigma (tuple), gravitational: G (float).
        box_bounds (tuple): The bounds of the simulation box. Needed for measuring the particle distances.
        boundary_conditions (str): The type of boundary conditions to apply. Options: "periodic", "reflective", "none".

    Returns:
        numpy.ndarray: The updated configuration after one time step.
    """
    # Extract current properties from the configuration
    configuration = data[timestep - 1]
    positions = configuration[params.properties["x"] : params.properties["z"] + 1, :]
    velocities = configuration[params.properties["vx"] : params.properties["vz"] + 1, :]
    forces = configuration[params.properties["fx"] : params.properties["fz"] + 1, :]
    masses = configuration[params.properties["mass"], :]

    # Update positions
    new_positions = positions + velocities * dt + 0.5 * forces * dt**2 / masses
    # Update forces based on the new positions as a helping step
    updated_configuration = np.copy(configuration)
    updated_configuration[params.properties["x"] : params.properties["z"] + 1, :] = (
        new_positions
    )
    # Update forces and hereby velocities
    new_force_matrix = force_matrix(
        updated_configuration,
        potential,
        potential_params,
        box_bounds,
        boundary_conditions,
    )
    new_forces = np.sum(new_force_matrix, axis=1).T
    new_velocities = velocities + 0.5 * (forces + new_forces) * dt / masses

    # Create new configuration
    new_configuration = np.copy(configuration)
    new_configuration[params.properties["x"] : params.properties["z"] + 1, :] = (
        new_positions
    )
    new_configuration[params.properties["vx"] : params.properties["vz"] + 1, :] = (
        new_velocities
    )
    new_configuration[params.properties["fx"] : params.properties["fz"] + 1, :] = (
        new_forces
    )

    return new_configuration
