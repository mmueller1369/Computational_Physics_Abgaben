"""
Author: Elias Jedam & Matthias Müller
Date: 09.05.2025

This file generates the initial conditions for the LJ fluid
"""

import numpy as np
import params

lj_system_initial_configuration = np.array(
    [
        range(0, params.nparticles**2, 1),  # Particle ID
        ["1"] * params.nparticles**2,  # Type
        np.zeros(params.nparticles**2),  # x position
        np.zeros(params.nparticles**2),  # y position
        np.zeros(params.nparticles**2),  # z position
        np.zeros(params.nparticles**2),  # x velocity
        np.zeros(params.nparticles**2),  # y velocity
        np.zeros(params.nparticles**2),  # z velocity
        np.zeros(params.nparticles**2),  # x force
        np.zeros(params.nparticles**2),  # y force
        np.zeros(params.nparticles**2),  # z force
        np.ones(params.nparticles**2) * params.mass,  # mass
        np.ones(params.nparticles**2),  # radius
    ],
    dtype=object,
)


def create_geometry(initial_config: np.ndarray) -> np.ndarray:
    """
    assume 2D system

    Args:
        initial_config (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    rho = params.nparticles**2 / params.box_bounds[0][1] / params.box_bounds[1][1]
    alat = 1 / np.sqrt(rho)
    # change x and y coordinates of the particles
    for i in range(len(initial_config[0])):
        initial_config[2][i] = (i % params.nparticles) * alat + alat / 2  # x position
        initial_config[3][i] = (i // params.nparticles) * alat + alat / 2  # y position

    # change radius of the particles
    initial_config[12] = np.ones(params.nparticles**2) * alat / 2 * 0.2

    return initial_config


def assign_random_velocities(initial_config: np.ndarray) -> np.ndarray:
    """
    Assign random velocities to the particles in the initial configuration.

    Args:
        initial_config (np.ndarray): The initial configuration of the particles.

    Returns:
        np.ndarray: The initial configuration with assigned random velocities.
    """
    # Assign random velocities
    for i in range(len(initial_config[0])):
        initial_config[5][i] = np.random.uniform(-1, 1)  # x velocity
        initial_config[6][i] = np.random.uniform(-1, 1)  # y velocity
    # substract the average velocity
    initial_config[5] -= np.mean(initial_config[5])
    initial_config[6] -= np.mean(initial_config[6])

    return initial_config


print("done")
