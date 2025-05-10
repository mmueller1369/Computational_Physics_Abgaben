import numpy as np
import params
from typing import Union


# General: Calculate the force between two particles using an arbitrary potential model.
def force_matrix(
    config: np.ndarray,
    potential: str,
    potential_params: Union[tuple, float],
    box_bounds: tuple,
    boundary_conditions: str,
) -> np.ndarray:
    """
    General: Calculate the force matrix for a given configuration and potential. Uses the symmetry of the force matrix to reduce computation time.

    Parameters:
        config (numpy.ndarray): The configuration of the system. Corresponds to the shape (properties, particles) and thus to an entry in the data array for a specific timestep.
        potential (str): The potential model to use. Options: "lj", "gravitational".
        potential_params (Union[tuple, float]): The parameters for the potential model; lj: epsilon and sigma (tuple), gravitational: G (float).
        box_bounds (tuple): The bounds of the simulation box. Needed for measuring the particle distances.
        boundary_conditions (str): The type of boundary conditions to apply. Options: "periodic", "reflective", "none".

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
            force = potential_func(
                prop1, prop2, potential_params, boundary_conditions, box_bounds
            )
            force_matrix[i, j] = force
            force_matrix[j, i] = -force  # Use symmetry of the force matrix

    return force_matrix


def effective_distance(
    prop1: np.ndarray, prop2: np.ndarray, box_bounds: tuple, boundary_conditions: str
) -> tuple:
    """
    Calculate the distance between two particles considering the specified periodic boundary conditions.

    Parameters:
        prop1 (numpy.ndarray): Properties of the first particle.
        prop2 (numpy.ndarray): Properties of the second particle.
        box_bounds (tuple): The bounds of the simulation box.
        boundary_conditions (str): The type of boundary conditions to apply.

    Returns:
        tuple: The distance vector and its norm.
    """
    r2, r1 = (
        prop2[params.properties["x"] : params.properties["z"] + 1],
        prop1[params.properties["x"] : params.properties["z"] + 1],
    )
    r = (r2 - r1).reshape(3, 1)

    if boundary_conditions == "periodic":
        box_lengths = np.ones(3) * box_bounds[0][1]
        r = (r2 - r1 + box_lengths / 2) % box_lengths - box_lengths / 2
    return r, np.linalg.norm(r)


# Definition of the explicit force functions
def lj_force(
    prop1: np.ndarray,
    prop2: np.ndarray,
    potential_params: tuple,
    boundary_conditions: str,
    box_bounds: tuple,
) -> np.ndarray:
    """
    Lennard-Jones potential force calculation between two particles.

    Parameters:
        prop1 (numpy.ndarray): Properties of the first particle.
        prop2 (numpy.ndarray): Properties of the second particle.
        potential_params (tuple): Parameters for the Lennard-Jones potential; epsilon and sigma.
        boundary_conditions (str): The type of boundary conditions to apply.
        box_bounds (tuple): The bounds of the simulation box.

    Returns:
        numpy.ndarray: The force vector between the two particles.
    """
    r, r_abs = effective_distance(prop1, prop2, box_bounds, boundary_conditions)
    epsilon, sigma = potential_params
    f = -(
        4
        * epsilon
        * ((12 * sigma**12 / r_abs**13) - (6 * sigma**6 / r_abs**7))
        * (r / r_abs)
    )
    return f.flatten()


def lj_cut_force(
    prop1: np.ndarray,
    prop2: np.ndarray,
    potential_params: tuple,
    boundary_conditions: str,
    box_bounds: tuple,
) -> np.ndarray:
    """
    Lennard-Jones potential force calculation with cutoff between two particles.

    Parameters:
        prop1 (numpy.ndarray): Properties of the first particle.
        prop2 (numpy.ndarray): Properties of the second particle.
        potential_params (tuple): Parameters for the Lennard-Jones potential; epsilon, sigma and cutoff.
        boundary_conditions (str): The type of boundary conditions to apply.
        box_bounds (tuple): The bounds of the simulation box.

    Returns:
        numpy.ndarray: The force vector between the two particles.
    """
    r, r_abs = effective_distance(prop1, prop2, box_bounds, boundary_conditions)
    epsilon, sigma, cutoff = potential_params
    if r_abs < cutoff:
        f = -(
            4
            * epsilon
            * ((12 * sigma**12 / r_abs**13) - (6 * sigma**6 / r_abs**7))
            * (r / r_abs)
        )
    else:
        f = np.zeros(3)
    return f.flatten()


def gravitational_force(
    prop1: np.ndarray,
    prop2: np.ndarray,
    potential_params: tuple,
    boundary_conditions: str,
    box_bounds: tuple,
) -> np.ndarray:
    """
    Gravitational force calculation between two particles.

    Parameters:
        prop1 (numpy.ndarray): Properties of the first particle.
        prop2 (numpy.ndarray): Properties of the second particle.
        potential_params (float): Gravitational constant.
        boundary_conditions (str): The type of boundary conditions to apply.
        box_bounds (tuple): The bounds of the simulation box.

    Returns:
        numpy.ndarray: The force vector between the two particles.
    """
    r, r_abs = effective_distance(prop1, prop2, box_bounds, boundary_conditions)
    m1, m2 = prop1[params.properties["mass"]], prop2[params.properties["mass"]]
    G = potential_params
    f = G * m1 * m2 / r_abs**2 * (r / r_abs)
    return f.flatten()


# Corresponding potential functions
def lj_potential(
    prop1: np.ndarray,
    prop2: np.ndarray,
    potential_params: tuple,
    boundary_conditions: str,
    box_bounds: tuple,
) -> np.ndarray:
    """
    Lennard-Jones potential calculation between two particles.

    Parameters:
        prop1 (numpy.ndarray): Properties of the first particle.
        prop2 (numpy.ndarray): Properties of the second particle.
        potential_params (tuple): Parameters for the Lennard-Jones potential; epsilon and sigma.
        boundary_conditions (str): The type of boundary conditions to apply.
        box_bounds (tuple): The bounds of the simulation box.

    Returns:
        float: The potential energy between the two particles.
    """
    r, r_abs = effective_distance(prop1, prop2, box_bounds, boundary_conditions)
    epsilon, sigma = potential_params
    V = 4 * epsilon * ((sigma / r_abs) ** 12 - (sigma / r_abs) ** 6)
    return V


def lj_cut_potential(
    prop1: np.ndarray,
    prop2: np.ndarray,
    potential_params: tuple,
    boundary_conditions: str,
    box_bounds: tuple,
) -> np.ndarray:
    """
    Lennard-Jones potential calculation with cutoff between two particles.

    Parameters:
        prop1 (numpy.ndarray): Properties of the first particle.
        prop2 (numpy.ndarray): Properties of the second particle.
        potential_params (tuple): Parameters for the Lennard-Jones potential; epsilon, sigma and cutoff.
        boundary_conditions (str): The type of boundary conditions to apply.
        box_bounds (tuple): The bounds of the simulation box.

    Returns:
        float: The potential energy between the two particles.
    """
    r, r_abs = effective_distance(prop1, prop2, box_bounds, boundary_conditions)
    epsilon, sigma, cutoff = potential_params
    if r_abs < cutoff:
        V = 4 * epsilon * (
            (sigma / r_abs) ** 12 - (sigma / r_abs) ** 6
        ) - 4 * epsilon * (cutoff**-12 - cutoff**-6)
    else:
        V = 0
    return V


def gravitational_potential(
    prop1: np.ndarray,
    prop2: np.ndarray,
    potential_params: tuple,
    boundary_conditions: str,
    box_bounds: tuple,
) -> np.ndarray:
    """
    Gravitational potential calculation between two particles.

    Parameters:
        prop1 (numpy.ndarray): Properties of the first particle.
        prop2 (numpy.ndarray): Properties of the second particle.
        potential_params (float): Gravitational constant.
        boundary_conditions (str): The type of boundary conditions to apply.
        box_bounds (tuple): The bounds of the simulation box.

    Returns:
        float: The potential energy between the two particles.
    """
    r, r_abs = effective_distance(prop1, prop2, box_bounds, boundary_conditions)
    m1, m2 = prop1[params.properties["mass"]], prop2[params.properties["mass"]]
    V = -potential_params * m1 * m2 / r_abs
    return V
