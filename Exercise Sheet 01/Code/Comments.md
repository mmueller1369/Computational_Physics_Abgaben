# Run the code

# main

It is necessary to have the same folder structure.
Exersice Sheet
| -> Code
| -> output
| -> provided
or manually change every filepath in the files.

## analysis

You can execute:

- center_of_mass.py
- energies.py
- plot_data.py
- compare_trajectory.py

# Data

## How to read in and use data

The data is stored in numpy arrays. Necessary for the calculations are  
mass  
positions  
velocity  
force
all of them are in a shape of (10,3) or mass in (10,1)

## How to store the data

Instead of writing into a file for each step. The data is stored in a numpy array, because I have  
enough RAM XD. After calculations the data is exported into a .npy. Then data can be used, e.g.
to create a readable Ovito file.

# Where can I find the key formula?

## Velocity-Verlet

In algorithm.py line 26 and 29.

## Verlet

In algorithm.py line 84. We dont need an equation for velocity.

## Euler

In algorithm.py line 118 and 121.

## Force and Potential

Line 31 in solar_system.py will calculate the force between two particles. The function system_force
in line 34 (solar_system.py) will calculate the total force due to interaction with other particle.
At the moment the code calculate the force for each field in the "matrix" even though the matrix is Skew-symmetric.
=> One can save speed here.
