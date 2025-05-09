# Simulation parameters
## Number of particles in the simulation
nparticles = 10
## Time step size measured in the units of the simulation
dt = 1
## Amount of time steps executed of the simulation
dt_max = 100
## Integrator algorithm used in the simulation; options: "velocity_verlet", "verlet", "euler"
integrator = "velocity_verlet"
## Potential model used in the simulation; options: "lj", "gravitational"
potential = "lj_cut"
## Parameters for the potential model; lj: epsilon and sigma, lj_cut: epsilon, sigma and cutoff (in the units of sigma), gravitational: G; type: list for lj, lj_cut, float for gravitational
potential_params = [1, 1, 2.5]
## Thermostat used in the simulation; options: "none", "berendsen", "nose-hoover"
thermostat = "none"
## Temperature of the system in the units of the simulation
T = 1.0
## Time step for thermostat updates
dt_thermostat = 1
## Boltzmann constant; used for thermostatting
kB = 0.0019849421
## Simulation box bounds
box_bounds = ((0, 10), (0, 10), (0, 10))
## Boundary conditions; options: "periodic", "reflective", "none"
boundary_conditions = "periodic"


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
## Name of the output/input file
filename = "file.dat"
## Directory where the file will be saved; if None, current directory is used
path = None
## Export data every dt_export steps
dt_export = 1
