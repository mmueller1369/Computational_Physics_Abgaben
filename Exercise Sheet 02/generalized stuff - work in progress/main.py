import numpy as np
from simulation_tools import particle, md_simulation
import params

# iterate for all integrators

for j, output_filename in enumerate(['output_velocity_verlet.dat', 'output_verlet.dat', 'output_euler.dat']):
    
    
    # initializiation of the system and creation of the planets (sun is counted as a planet too)
    planets = [particle(np.zeros(3), np.zeros(3), np.zeros(3), 0, 'type') for _ in range(params.nparticles)] 
    
    
    # attribute the initial conditions to the planets; not included in modules as this is highly specific for the problem
    file_mass = open('mass.dat', 'r')
    for i, lines_mass in enumerate(file_mass.readlines()[3:]):
        planet = planets[i]
        mass = float(lines_mass.split()[0])
        name = lines_mass.split()[-1]
        planet.m = mass
        planet.name = name
    file_mass.close()
    
    file_vectors = open('planets_April_22_2025.dat', 'r')
    for i, lines_vectors in enumerate(file_vectors.readlines()[8:-2]):
        planet = planets[i//2 + 1] # two lines per planet (i//2), sun is skipped (+ 1)
        if i%2 == 0: # first line = position
            vector = np.array(lines_vectors.split()[:3], dtype=float)
            planet.r = vector
        if i%2 == 1: # second line = velocity
            vector = np.array(lines_vectors.split()[:3], dtype=float)
            planet.v = vector
    file_vectors.close()


    # execution
    md_simulation(planets, j+1, 1, output_filename)