import numpy as np
import params
from forces import lj_potential, lj_cut_potential, gravitational_potential


def compute_temperature(data: np.ndarray, timestep: int) -> float:
    """
    Compute the average temperature from the given data.

    Args:
        data (np.ndarray): The simulation data.
        timestep (int): The current timestep.

    Returns:
        float: The average temperature.
    """
    nparticles = data[timestep].shape[1]
    kinetic_energy = compute_ke(data, timestep)
    temperature = (2 * kinetic_energy) / (3 * nparticles * params.k_B)
    return temperature


def compute_ke(data: np.ndarray, timestep: int) -> float:
    """
    Compute the kinetic energy from the given data.

    Args:
        data (np.ndarray): The simulation data.
        timestep (int): The current timestep.

    Returns:
        float: The kinetic energy.
    """
    configuration = data[timestep]
    velocities = configuration[params.properties["vx"] : params.properties["vz"] + 1, :]
    masses = configuration[params.properties["mass"], :]
    kinetic_energy = 0.5 * np.sum(masses * np.sum(velocities**2, axis=0))
    return kinetic_energy


def compute_pe(
    data: np.ndarray, timestep: int, potential: str, potential_params: float
) -> float:
    """
    Compute the potential energy from the given data.

    Args:
        data (np.ndarray): The simulation data.
        timestep (int): The current timestep.
        potential (str): The potential model to use.
        potential_params (float): The parameters for the potential model.

    Returns:
        float: The potential energy.
    """
    configuration = data[timestep]
    nparticles = configuration.shape[1]
    potential_func = globals()[f"{potential}_potential"]
    potential_energy = 0.0
    for i in range(nparticles):
        prop_i = configuration[:, i]
        for j in range(i + 1, nparticles):
            prop_j = configuration[:, j]
            potential_energy += potential_func(prop_i, prop_j, potential_params)
    return potential_energy


def compute_e(
    data: np.ndarray, timestep: int, potential: str, potential_params: float
) -> float:
    """
    Compute the total energy from the given data, using the computes for kinetic and potential energy.

    Args:
        data (np.ndarray): The simulation data.
        timestep (int): The current timestep.
        potential (str): The potential model to use.
        potential_params (float): The parameters for the potential model.

    Returns:
        float: The total energy.
    """
    ke = compute_ke(data, timestep)
    pe = compute_pe(data, timestep, potential, potential_params)
    return ke + pe
