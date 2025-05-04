"""
Author: Elias Jedam & Matthias MÃ¼ller
Date: 04/05/2025

Energy Calculation
"""

import numpy as np
import os
import my_io
from tqdm import tqdm
import matplotlib.pyplot as plt

BASIS_PFAD = os.path.dirname(os.path.dirname(__file__))


def compute_pe(simulation="verlet") -> np.ndarray:
    """
    Computes the potential energy of the system at each timestep from the file
    Returns the potential energy at each timestep
    -----------
    Parameters:
    simulation: data from which simulation should be used
    """
    filename_pos = os.path.join(BASIS_PFAD, "output", f"positions_{simulation}.npy")
    positions = np.load(filename_pos)
    mass = my_io.read_masses(
        "mass.dat"
    )  # in units of 10^29 kg sun is first element and pluto last
    mass = mass.reshape(-1, 1)  # reshape mass to be a column vector
    
    pe = np.zeros(365000)
    G = 1.488136e-5
    # iteration uses the fact that it is a pair potential and thus symmetrical
    for i in tqdm(range(10)):
        for j in range(i):
            pos_i = positions[::,i,::]
            pos_j = positions[::,j,::]
            mass_i = mass[i][0]
            mass_j = mass[j][0]
            distance_vector = pos_i - pos_j
            distance = np.array([np.linalg.norm(dist) for dist in distance_vector])
            pe_ij = 2 * (- G * mass_i * mass_j / distance)
            pe += pe_ij
    return pe

def compute_ke(simulation="verlet") -> np.ndarray:
    """
    Computes the kinetic energy of the system at each timestep from the file
    Returns the kinetic energy at each timestep
    -----------
    Parameters:
    simulation: data from which simulation should be used
    """
    filename_vel = os.path.join(BASIS_PFAD, "output", f"velocities_{simulation}.npy")
    velocities = np.load(filename_vel)
    mass = my_io.read_masses(
        "mass.dat"
    )  # in units of 10^29 kg sun is first element and pluto last
    mass = mass.reshape(-1, 1)  # reshape mass to be a column vector
    
    ke = np.zeros(365000)
    # iteration uses the fact that it is a pair potential and thus symmetrical
    for i in tqdm(range(10)):
        velocities_i = velocities[::,i,::]
        vel_i = np.array([np.linalg.norm(vel) for vel in velocities_i])
        mass_i = mass[i]
        ke += 1 / 2 * mass_i * vel_i * vel_i
    return ke

def compute_e(simulation="verlet") -> np.ndarray:
    """
    Computes the total energy of the system at each timestep from the file
    Returns the total energy at each timestep
    -----------
    Parameters:
    simulation: data from which simulation should be used
    """
    ke = compute_ke(simulation)
    pe = compute_pe(simulation)
    return ke + pe
    

etot_vv = compute_e('vel_verlet')
etot_v = compute_e('verlet')
etot_e = compute_e('euler')
plt.plot(etot_vv, label = 'Velocity Verlet')
plt.plot(etot_v, label = 'Verlet')
plt.plot(etot_e, label = 'Euler')
plt.xlabel('Timestep')
plt.ylabel(r'$E_{tot}$')
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(BASIS_PFAD, "output", f"energy_plot.png")
)