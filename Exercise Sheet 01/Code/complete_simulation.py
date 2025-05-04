"""
Author: Elias Jedam & Matthias Müller
Date02/05/2025
This file contains function to run the simulation and write the output data into
a file with one function.
"""

import os
import numpy as np
from algorithm import velocity_verlet, verlet, euler
import data_storage as ds
import my_io


def run_velocity_verlet(
    mass,
    positions,
    velocities,
    forces,
    radii_au,
    output_filename="vel_verlet.npy",
    lammps_file="vel_verlet.dat",
):
    """_summary_

    Args:
        mass (_type_): _description_
        positions (_type_): _description_
        velocities (_type_): _description_
        forces (_type_): _description_
        radii_au (_type_): _description_
        output_filename (str, optional): _description_. Defaults to "vel_verlet.npy".
        lammps_file (str, optional): _description_. Defaults to "vel_verlet.dat".
    """

    # start simulation
    velocity_verlet(
        mass,
        positions,
        velocities,
        forces,
        particles=ds.number_particles,
        dt=ds.dt,
        steps=ds.max_steps,
        output_filename=output_filename,
    )
    my_io.write_lammps(
        "positions_" + output_filename,
        lammps_file,
        ds.number_particles,
        ds.box_bounds,
        mass=mass,
        radius=radii_au,
    )


def run_verlet(
    mass,
    positions0,
    forces,
    radii_au,
    velocities,
    output_filename="verlet.npy",
    lammps_file="verlet.dat",
):
    """_summary_

    Args:
        mass (_type_): _description_
        positions0 (_type_): _description_
        forces (_type_): _description_
        radii_au (_type_): _description_
        velocities (_type_): _description_
        output_filename (str, optional): _description_. Defaults to "verlet.npy".
        lammps_file (str, optional): _description_. Defaults to "verlet.dat".
    """
    # first I need a second set of positions to start the simulation
    # so i will use the first two positions from the velocity verlet simulation
    print(
        "I need to start a short velocity verlet simulation to get the second position"
    )
    velocity_verlet(
        mass,
        positions0,
        velocities,
        forces,
        particles=ds.number_particles,
        dt=ds.dt,
        steps=2,
        output_filename="help.npy",
    )
    basis_pfad = os.path.dirname(os.path.dirname(__file__))
    filename = os.path.join(basis_pfad, "output", "positions_help.npy")
    positions = np.load(filename)

    verlet(
        mass,
        positions[0],
        positions[1],
        velocities,
        forces,
        particles=ds.number_particles,
        dt=ds.dt,
        steps=ds.max_steps,
        output_filename=output_filename,
    )

    my_io.write_lammps(
        "positions_" + output_filename,
        lammps_file,
        ds.number_particles,
        ds.box_bounds,
        mass=mass,
        radius=radii_au,
    )


def run_euler(
    mass,
    positions,
    velocities,
    forces,
    radii_au,
    output_filename="euler.npy",
    lammps_file="euler.dat",
):
    """_summary_

    Args:
        mass (_type_): _description_
        positions (_type_): _description_
        velocities (_type_): _description_
        forces (_type_): _description_
        radii_au (_type_): _description_
        output_filename (str, optional): _description_. Defaults to "euler.npy".
        lammps_file (str, optional): _description_. Defaults to "euler.dat".
    """

    # start simulation
    euler(
        mass,
        positions,
        velocities,
        forces,
        particles=ds.number_particles,
        dt=ds.dt,
        steps=ds.max_steps,
        output_filename=output_filename,
    )
    my_io.write_lammps(
        "positions_" + output_filename,
        lammps_file,
        ds.number_particles,
        ds.box_bounds,
        mass=mass,
        radius=radii_au,
    )
