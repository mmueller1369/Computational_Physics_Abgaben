import numpy as np
import params
from typing import Union


# General: Calculate the force between two particles using an arbitrary potential model.
def force_matrix(
    config: np.ndarray,
    potential: str,
    potential_params: Union[tuple, float],
) -> np.ndarray:
    """
    General: Calculate the force matrix for a given configuration and potential. Uses the symmetry of the force matrix to reduce computation time.

    Parameters:
        config (numpy.ndarray): The configuration of the system. Corresponds to the shape (properties, particles) and thus to an entry in the data array for a specific timestep.
        potential (str): The potential model to use. Options: "lj", "gravitational".
        potential_params (Union[tuple, float]): The parameters for the potential model; lj: epsilon and sigma (tuple), gravitational: G (float).

    Returns:
        numpy.ndarray: The force matrix; shape (particles, particles, 3).
    """
    try:
        potential_func = globals()[f"{potential}_force"]
    except KeyError:
        raise ValueError(f"Unknown potential: {potential}")

    force_matrix = np.zeros((config.shape[1], config.shape[1], 3))
    for i in range(config.shape[1]):
        for j in range(i + 1, config.shape[1]):
            prop1 = config[:, i]
            prop2 = config[:, j]
            force = potential_func(prop1, prop2, potential_params)
            force_matrix[i, j] = force
            force_matrix[j, i] = -force  # Use symmetry of the force matrix

    return force_matrix


# Definition of the explicit force functions
def lj_force(
    prop1: np.ndarray, prop2: np.ndarray, potential_params: tuple
) -> np.ndarray:
    """
    Lennard-Jones potential force calculation between two particles.

    Parameters:
        prop1 (numpy.ndarray): Properties of the first particle.
        prop2 (numpy.ndarray): Properties of the second particle.
        potential_params (tuple): Parameters for the Lennard-Jones potential; epsilon and sigma.

    Returns:
        numpy.ndarray: The force vector between the two particles.
    """
    r2, r1 = (
        prop2[params.properties["x"] : params.properties["z"] + 1],
        prop1[params.properties["x"] : params.properties["z"] + 1],
    )
    r = (r2 - r1).reshape(3, 1)
    r_abs = np.linalg.norm(r)
    epsilon, sigma = potential_params
    f = (
        4
        * epsilon
        * ((12 * sigma**12 / r_abs**13) - (6 * sigma**6 / r_abs**7))
        * (r / r_abs)
    )
    return f.flatten()


def gravitational_force(
    prop1: np.ndarray, prop2: np.ndarray, potential_params: float
) -> np.ndarray:
    """
    Gravitational force calculation between two particles.

    Parameters:
        prop1 (numpy.ndarray): Properties of the first particle.
        prop2 (numpy.ndarray): Properties of the second particle.
        potential_params (float): Gravitational constant.

    Returns:
        numpy.ndarray: The force vector between the two particles.
    """
    r2, r1 = (
        prop2[params.properties["x"] : params.properties["z"] + 1],
        prop1[params.properties["x"] : params.properties["z"] + 1],
    )
    m1, m2 = prop1[params.properties["mass"]], prop2[params.properties["mass"]]
    r = (r2 - r1).reshape(3, 1)
    r_abs = np.linalg.norm(r)
    G = potential_params
    f = G * m1 * m2 / r_abs**2 * (r / r_abs)
    return f.flatten()


# Corresponding potential functions
def lj_potential(
    prop1: np.ndarray, prop2: np.ndarray, potential_params: tuple
) -> float:
    """
    Lennard-Jones potential calculation between two particles.

    Parameters:
        prop1 (numpy.ndarray): Properties of the first particle.
        prop2 (numpy.ndarray): Properties of the second particle.
        potential_params (tuple): Parameters for the Lennard-Jones potential; epsilon and sigma.

    Returns:
        float: The potential energy between the two particles.
    """
    r2, r1 = (
        prop2[params.properties["x"] : params.properties["z"] + 1],
        prop1[params.properties["x"] : params.properties["z"] + 1],
    )
    r = (r2 - r1).reshape(3, 1)
    r_abs = np.linalg.norm(r)
    epsilon, sigma = potential_params
    V = 4 * epsilon * ((sigma / r_abs) ** 12 - (sigma / r_abs) ** 6)
    return V


def gravitational_potential(
    prop1: np.ndarray, prop2: np.ndarray, potential_params: float
) -> float:
    """
    Gravitational potential calculation between two particles.

    Parameters:
        prop1 (numpy.ndarray): Properties of the first particle.
        prop2 (numpy.ndarray): Properties of the second particle.
        potential_params (float): Gravitational constant.

    Returns:
        float: The potential energy between the two particles.
    """
    r2, r1 = (
        prop2[params.properties["x"] : params.properties["z"] + 1],
        prop1[params.properties["x"] : params.properties["z"] + 1],
    )
    m1, m2 = prop1[params.properties["mass"]], prop2[params.properties["mass"]]
    r = (r2 - r1).reshape(3, 1)
    r_abs = np.linalg.norm(r)
    V = -potential_params * m1 * m2 / r_abs
    return V
