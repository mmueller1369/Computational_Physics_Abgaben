# global units, properties etc.
## Number of particles in the simulation
nparticles = 10
## Time step size measured in the units of the simulation
dt = 0
## Amount of time steps executed of the simulation
dt_max = 0


# data structure etc.
## Dictionary mapping property names to their indices in the data array
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
    "cluster_id": 13,
    "charge": 14,
}
## Name of the output/input file
filename = "file.dat"
## Directory where the file will be saved; if None, current directory is used
path = None
## Export data every dt_export steps
dt_export = 1


# Simulation parameters
## Integrator algorithm used in the simulation; options: "velocity_verlet", "verlet", "euler"
integrator = "velocity_verlet"
## Force used in the simulation; options: "lj", "grav"
force = "lj"
## Thermostat used in the simulation; options: "none", "berendsen", "nose-hoover"
thermostat = "none"
## Simulation box bounds
box_bounds = ((0, 10), (0, 10), (0, 10))
