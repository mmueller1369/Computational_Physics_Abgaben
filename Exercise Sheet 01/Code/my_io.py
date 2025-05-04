"""
Author: Elias Jedam & Matthias
Date: 30/04/2025

Diese Klasse soll Daten des .sec Formats einelesen könnten.
"""

import os
import numpy as np


def read_masses(filename):
    basis_pfad = os.path.dirname(os.path.dirname(__file__))
    filename = os.path.join(basis_pfad, "provided", filename)
    masses = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    mass_str = line.split()[0]
                    mass = float(mass_str)
                    masses.append(mass)
                except (IndexError, ValueError):
                    continue  # Skip lines that don't contain valid float
    return np.array(masses)


def read_positions_velocities_with_sun(filename):
    basis_pfad = os.path.dirname(os.path.dirname(__file__))
    filename = os.path.join(basis_pfad, "provided", filename)
    positions = [np.zeros(3)]  # Sun at origin
    velocities = [np.zeros(3)]  # Sun stationary

    with open(filename, "r") as file:
        lines = file.readlines()

    # Filter: keep only non-comment lines with 3 numeric values
    data_lines = [
        line
        for line in lines
        if not line.strip().startswith("!%")  # Exclude comment lines
    ]

    # Remove any extra comment at the end of the lines
    data_lines = [
        line.strip().split(" !%")[0]  # Strip out the comment part (after ' !%')
        for line in data_lines
        if line.strip()  # Remove empty lines
    ]

    # Process position and velocity pairs
    for i, line in enumerate(data_lines):
        parts = line.split()
        if len(parts) < 3:
            continue  # skip malformed lines
        vector = np.array(parts[:3], dtype=float)
        if i % 2 == 0:
            positions.append(vector)
        else:
            velocities.append(vector)

    return np.array(positions), np.array(velocities)


def write_lammps(pos_file, output_file, num_particles, box_bounds, mass, radius):
    basis_pfad = os.path.dirname(os.path.dirname(__file__))
    filename = os.path.join(basis_pfad, "output", pos_file)
    output_file = os.path.join(basis_pfad, "output", output_file)
    positions = np.load(filename)
    velocities = np.load(filename.replace("positions", "velocities"))
    planet_list = [
        "Sun",
        "Mercury",
        "Venus",
        "Earth",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
        "Pluto",
    ]
    with open(output_file, "w", encoding="utf-8") as file:
        for t, r in enumerate(positions):
            file.write("ITEM: TIMESTEP\n")
            file.write(f"{t}\n")
            file.write("ITEM: NUMBER OF ATOMS\n")
            file.write(f"{num_particles}\n")
            file.write("ITEM: BOX BOUNDS\n")
            file.write(f"{box_bounds[0][0]} {box_bounds[0][1]} xlo xhi\n")
            file.write(f"{box_bounds[1][0]} {box_bounds[1][1]} ylo yhi\n")
            file.write(f"{box_bounds[2][0]} {box_bounds[2][1]} zlo zhi\n")
            file.write("ITEM: ATOMS id type radius x y z vx vy vz\n")
            for i, planet in enumerate(r):
                file.write(
                    (
                        f"{i} {planet_list[i]}"
                        f" {0.1}"  # {mass[i][0]:.6f}"
                        f" {planet[0]:.6e} {planet[1]:.6e} {planet[2]:.6e}"
                        f" {velocities[t][i][0]:.6e} {velocities[t][i][1]:.6e} {velocities[t][i][2]:.6e}\n"
                    )
                )
