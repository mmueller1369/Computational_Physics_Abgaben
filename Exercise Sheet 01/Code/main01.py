"""
Author: Elias Jedam & Matthias Müller
Date: 30/04/2025

Main Datei
"""

import time
import numpy as np
from complete_simulation import run_velocity_verlet, run_verlet, run_euler
import data_storage as ds
import my_io

if __name__ == "__main__":
    print("Starting the program...")
    start_time = time.time()

    # init particles
    mass = my_io.read_masses(
        "mass.dat"
    )  # in units of 10^29 kg sun is first element and plut last
    positions, velocities = my_io.read_positions_velocities_with_sun(
        "planets_April_22_2025.dat"
    )
    mass = mass.reshape(-1, 1)  # reshape mass to be a column vector
    forces = np.zeros((ds.number_particles, 3))
    # Mean planetary radii in kilometers but for now not used, because I cant see so
    # small planets in ovito
    radii_km = np.array(
        [
            695700.0,  # Sun
            2439.7,  # Mercury
            6051.8,  # Venus
            6371.0,  # Earth
            3389.5,  # Mars
            69911.0,  # Jupiter
            58232.0,  # Saturn
            25362.0,  # Uranus
            24622.0,  # Neptune
            1188.3,  # Pluto (dwarf planet)
        ]
    )

    # Convert km to AU
    radii_au = radii_km / 1.495978707e8

    # start velocity verlet simulation
    run_velocity_verlet(mass, positions, velocities, forces, radii_au)
    # start verlet simulation
    run_verlet(mass, positions, forces, radii_au, velocities)
    # run euler simulation
    run_euler(mass, positions, velocities, forces, radii_au)

    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.2f} seconds")
