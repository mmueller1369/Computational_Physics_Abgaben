import numpy as np
import params
from forces import gravitation_potential


def compute_distance(particle_1, particle_2) -> np.ndarray:
    """
    Computes the distance verctor between two given particles at a given timestep
    -----------
    Parameters:
    particle_1, 2: corresponding particles (from particle class)
    """
    return particle_2.r - particle_1.r


def compute_com(particles: list) -> np.ndarray:
    """
    Computes the COM vector of a list of particles at a given timestep
    -----------
    Parameters:
    particles: corresponding particles (from particle class)
    """
    total_mass = np.sum(np.array([particle.m for particle in particles]))
    com = (
        np.sum(np.array([particle.m * particle.r for particle in particles]))
        / total_mass
    )
    return com


def compute_ke(particles: list) -> float:
    """
    Computes the kinetic energy of a list of particles at a given timestep
    -----------
    Parameters:
    particles: corresponding particles (from particle class)
    """
    ke = np.sum(
        np.array(
            [
                1 / 2 * particle.m * np.dot(particle.v, particle.v)
                for particle in particles
            ]
        )
    )
    return ke


def compute_pe(particles: list, potential_number: int) -> float:
    """
    Computes the potential energy of a list of particles at a given timestep; uses the fact that it is a pair potential
    -----------
    Parameters:
    particles: corresponding particles (from particle class)
    potential_number: 1 = gravitation
    """
    pe = 0
    if potential_number == 1:
        potential = gravitation_potential
    for i in range(params.nparticles):
        for j in range(i):
            particle1 = particles[i]
            particle2 = particles[j]
            pe += 2 * potential(particle1, particle2)
    return pe


def compute_e(particles: list, potential_number: int) -> float:
    """
    Computes the total energy of a list of particles at a given timestep; uses compute_pe, compute_ke
    -----------
    Parameters:
    particles: corresponding particles (from particle class)
    potential_number: 1 = gravitation
    """
    pe = compute_pe(particles, potential_number)
    ke = compute_ke(particles)
    return pe + ke
