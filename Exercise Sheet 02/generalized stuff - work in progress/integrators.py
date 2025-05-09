import numpy as np
import params
from forces import force_matrix        

def velocity_verlet(planets: list, force_number: int) -> None:
    """
    Performs an MD step using the Velocity Verlet algorithm, updates all properties of the entire system.
    -----------
    Parameters:
    planets: list containing the planets
    force_number: 1 = gravitation
    """
    # updating the positions
    for planet in planets:
        planet.r = planet.r + planet.v * params.dt + planet.f * params.dt * params.dt / 2 / planet.m
    # calculating the new grav_matr
    force_matrix_new = force_matrix(planets, force_number)
    
    for i, planet in enumerate(planets):
        # calculating the effective force for each planet
        force_new_x = np.sum(force_matrix_new[0,::,i])
        force_new_y = np.sum(force_matrix_new[1,::,i])
        force_new_z = np.sum(force_matrix_new[2,::,i])
        force_new = np.array([force_new_x, force_new_y, force_new_z])
        # updating the velocity and the force
        planet.v = planet.v + (planet.f + force_new) * params.dt / 2 / planet.m
        planet.f = force_new
        
def verlet(planets: list, force_number: int, earlier_positions: np.ndarray) -> np.ndarray:
    """
    Performs an MD step using the Verlet algorithm, updates all properties of the entire system.
    Returns the current positions (which will be earlier positions in the following timestep) as an array (nparticles x 3)
    -----------
    Parameters:
    planets: list containing the planets
    force_number: 1 = gravitation
    earlier_positions: needs an array of the earlier positions as these aren't part of the properties of each particle
    """
    new_earlier_positions = np.array([planet.r for planet in planets])
    # updating the positions and velocities
    for planet, earlier_position in zip(planets, earlier_positions):
        planet.r = 2 * planet.r - earlier_position + planet.f * params.dt * params.dt / 2 / planet.m
        planet.v = (planet.r - earlier_positions) / (2 * params.dt)
    # updating the forces
    force_matrix_new = force_matrix(planets, force_number)
    for i, planet in enumerate(planets):
        force_new_x = np.sum(force_matrix_new[0,::,i])
        force_new_y = np.sum(force_matrix_new[1,::,i])
        force_new_z = np.sum(force_matrix_new[2,::,i])
        force_new = np.array([force_new_x, force_new_y, force_new_z])
        planet.f = force_new
    
    return new_earlier_positions 

def euler(planets: list, force_number: int) -> None:
    """
    Performs an MD step using the Euler algorithm, updates all properties of the entire system.
    -----------
    Parameters:
    planets: list containing the planets
    force_number: 1 = gravitation
    """
    # updating the positions and velocities
    for planet in planets:
        planet.r = planet.r + planet.v * params.dt + planet.f * params.dt * params.dt / 2 / planet.m
        planet.v = planet.v + params.dt / planet.m * planet.f
    # updating the forces
    force_matrix_new = force_matrix(planets, force_number)
    for i, planet in enumerate(planets):
        force_new_x = np.sum(force_matrix_new[0,::,i])
        force_new_y = np.sum(force_matrix_new[1,::,i])
        force_new_z = np.sum(force_matrix_new[2,::,i])
        force_new = np.array([force_new_x, force_new_y, force_new_z])
        planet.f = force_new