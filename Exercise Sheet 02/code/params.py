import os
import numpy as np

# Simulation parameters
## Number of particles in the simulation
nparticles = 5
dt = 1e-15  # in seconds
dt_max = 5000
epsilon = 0.297741315 * 4184  # umrechnung in joule/mol
sigma = 0.188 * 1e-9  # in m
cutoff = 2.5 * sigma
kB = 0.0019849421 * 4184  # umrechnung in joule/mol/K
T = 300
mass = 39.95 * 1e-3  # in kg
potential_params = [epsilon, sigma, cutoff]
dt_thermostat = 10  # nach schritten updaten
potential = "lj_cut"
integrator = "velocity_verlet"
boundary_conditions = "periodic"
thermostat = "none"


## Simulation box bounds
L = 2 * sigma * nparticles
box_bounds = ((0, L), (0, L), (0, L))
# data structure etc.
## Dictionary mapping property names to their indices in the data array; possible properties: pID, type, x, y, z, vx, vy, vz, fx, fy, fz, mass, radius, cluster_id, charge
properties = {
    "pID": 0,
    "type": 1,
    "x": 2,
    "y": 3,
    "z": 4,
    "vx": 5,
    "vy": 6,
    "vz": 7,
    "fx": 8,
    "fy": 9,
    "fz": 10,
    "mass": 11,
    "radius": 12,
}
filename = "output.dat"  # standard name for output file
## Directory where the file will be saved; if None, current directory is used
path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
## Export data every dt_export steps
dt_export = 1
