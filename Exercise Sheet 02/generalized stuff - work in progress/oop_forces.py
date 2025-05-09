import numpy as np
import params


# general:
def force_matrix(planets: list, force_number: int) -> np.ndarray:
    """
    Calculates the forces between all planets, reducing computation power by using the fact that they are pairwise;
    Returns a (3 x nplanets x nplanets)-matrix in the format (direction [x,y,z], planet acting on, planet reacting to; see PDF)
    -----------
    Parameters:
    planets: list containing the planets
    force_number: 1 = gravitation
    """
    if force_number == 1:
        force_function = gravitation
    for_matr = np.zeros((3, params.nparticles, params.nparticles))
    for i in range(params.nparticles):
        for j in range(i):
            force = force_function(planets[i], planets[j])
            for_matr[::, i, j] = -force
            for_matr[::, j, i] = force
    return for_matr


# explicit forces:
def gravitation(particle1, particle2, G=1.488136e-5) -> np.ndarray:
    """
    Calculates the gravitational force between two planets;
    Returns the 3d force vector
    -----------
    Parameters:
    particle1,2: particles/planets
    """
    distance = particle2.r - particle1.r
    abs_distance = np.linalg.norm(distance)
    force = G * particle1.m * particle2.m / abs_distance**3 * distance
    return force
