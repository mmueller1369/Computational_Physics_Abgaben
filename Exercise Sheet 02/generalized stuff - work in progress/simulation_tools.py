import numpy as np
import params
from integrators import velocity_verlet, verlet, euler
import os
from numba import njit
from tqdm import tqdm

class particle:
    """
    Parameters:
    r: position
    v: velocity
    f: force
    m: mass
    name: name which will be used as 'type' later on
    """
    def __init__(self, r: np.ndarray, v: np.ndarray, f: np.ndarray, m: float, name: str):
        self.r = r
        self.v = v
        self.f = f
        self.m = m
        self.name = name
        
        
def write_output(output_file, timestep: int, planets: list, box_bounds: list, output_mode: str = params.output_mode) -> None:
    """
    Append the output file in the wanted format.
    -----------
    Parameters:
    timestep: integer number of the current timestep (NOT corresponding to the system time, would have to be scaled by dt = 0.5)
    planets: list of the planets (objects) which each includes the information about its position etc.
    box_bounds : box bounds
    mode: which parameters are to be exported
    """    
    #general stuff
    output_file.write("ITEM: TIMESTEP\n")
    output_file.write(f"{timestep}\n")
    output_file.write("ITEM: NUMBER OF ATOMS\n")
    output_file.write(f"{params.nparticles}\n")
    output_file.write("ITEM: BOX BOUNDS\n")
    directions = [['xlo', 'xhi'], ['zlo', 'zhi'], ['zlo', 'zhi']]
    for bounds, direction in zip(box_bounds, directions):
        output_file.write(f"{bounds[0]} {bounds[1]} {direction[0]} {direction[1]}\n")
        
    #trajectories
    if output_mode == 'r':
        output_file.write("ITEM: ATOMS id type x y z\n")
        for i, planet in enumerate(planets):
            name = planet.name
            x, y, z = planet.r
            output_file.write(f"{i+1} {name} {x:.6f} {y:.6f} {z:.6f}\n")
    if output_mode =='rv':
        output_file.write("ITEM: ATOMS id type x y z vx vy vz\n")
        for i, planet in enumerate(planets):
            name = planet.name
            x, y, z = planet.r
            vx, vy, vz = planet.v
            output_file.write(f"{i+1} {name} {x:.6f} {y:.6f} {z:.6f} {vx:.6f} {vy:.6f} {vz:.6f}\n")
        


# @njit
def md_simulation(planets: list, integrator: int, force_number: int, output_filename: str = params.output_filename) -> None:
    """
    Performs the simulation and creates an output file.
    -----------
    Parameters:
    planets: list containing the planets
    integrator: number of the used integrator; 1 = Velocity Verlet, 2 = Verlet, 3 = Euler
    force_number: number of the used force; 1 = Gravitation
    output_filename: name of the ouput, originally taken from params
    """
    #testing whether file already exists
    if os.path.exists(output_filename):
        answer = input(f"File '{output_filename}' already exists; do you want to overwrite? (y/n): ")
        if answer == "y":
            output_file = open(output_filename, "w")
        else:
            print("Simulation aborted.")
            return None
    else:
        output_file = open(output_filename, "w")
    # output_file = open(output_filename, "w") #skip for fast debugging
    #execution
    with tqdm(total=params.tmax) as progress:
        timestep = 0
        if integrator == 2:
            earlier_positions = np.array([planet.r - planet.f * params.dt for planet in planets])
            while timestep < params.tmax:
                earlier_positions = verlet(planets, force_number, earlier_positions)
                if timestep%params.tsave == 0:
                    write_output(output_file, timestep, planets, params.box_bounds)
                timestep += 1
                progress.update(1)
        else:
            if integrator == 1:
                integrator_func = velocity_verlet
            if integrator == 3:
                integrator_func = euler
            while timestep < params.tmax:
                integrator_func(planets, force_number)
                if timestep%params.tsave == 0:
                    write_output(output_file, timestep, planets, params.box_bounds)
                timestep += 1
                progress.update(1)
    output_file.close()