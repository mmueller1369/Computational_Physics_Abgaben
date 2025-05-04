"""
Author: Elias Jedam & Matthias Müller
Date: 02/05/2025

This file contains the functions necessary for the simulation of the planets.

"""

import numpy as np
import data_storage as ds


def force(pos1, pos2, mass1, mass2):
    """
    Calculate the gravitational force between two bodies.

    Args:
        pos1 (np.ndarray): Position of the first body.
        pos2 (np.ndarray): Position of the second body.
        mass1 (float): Mass of the first body.
        mass2 (float): Mass of the second body.

    Returns:
        np.ndarray: Gravitational force vector acting on the first body.
    """
    G = 6.67408e-11  # Gravitational constant in m^3 kg^-1 s^-2
    G = G * 1e29 * 86400**2 * (1 / 1.49597870691e11) ** 3
    # adjusted G due to the units of the mass, velocity and distance
    r = pos1 - pos2
    r_abs = np.linalg.norm(r)
    return -G * mass1 * mass2 * r / r_abs**3


def system_forces(masses, positions):
    """
    Updates the force on each particle considering every particle in the system.
    Args:
        masses (np.ndarray): Array of masses of each particle.
        positions (np.ndarray): Array of positions of the particles.
        forces (np.ndarray): Array to store the updated forces for each particle.
    Returns:
        forces (np.ndarray): Updated forces
    """
    forces = np.zeros((ds.number_particles, 3))
    for i in range(ds.number_particles):
        pl_force = np.zeros(3, dtype=np.float64)
        for j in range(ds.number_particles):
            if i != j:
                mass1 = masses[i]
                mass2 = masses[j]
                pos1 = positions[i]
                pos2 = positions[j]
                force_vector = force(pos1, pos2, mass1, mass2)
                pl_force += force_vector
        forces[i] = pl_force
    return forces


def potential_energy(mass, positions):
    """_summary_

    Args:
        mass (_type_): _description_
        positions (_type_): _description_

    Returns:
        _type_: _description_
    """
    G = 6.67408e-11  # Gravitational constant in m^3 kg^-1 s^-2
    G = G * 1e29 * 86400**2 * (1 / 1.49597870691e11) ** 3
    # adjusted G due to the units of the mass, velocity and distance
    epot = 0.0
    for i, planet in enumerate(positions):
        for j, other_planets in enumerate(positions):
            if i != j:  # avoid self-interaction and division by zero
                r = planet - other_planets
                r_abs = np.linalg.norm(r)
                epot += -G * mass * mass / r_abs
    return epot


def system_energy(mass, positions, velocities):
    """_summary_

    Args:
        mass (_type_): _description_
        positions (_type_): _description_
        velocities (_type_): _description_

    Returns:
        _type_: _description_
    """
    K = 1 / 2 * mass * np.linalg.norm(velocities) ** 2
    # caclulate potential energy
    U = potential_energy(mass, positions)
    energy = np.sum(K + U)
    return energy
