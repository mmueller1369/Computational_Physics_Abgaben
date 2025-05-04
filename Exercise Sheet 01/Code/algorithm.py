"""
Auhtor: Elias Jedam & Matthias Müller
DAte: 02/05/2025

This file contains the algorithm for the simulation of the planets.
"""

import os
from solar_system import system_forces
import numpy as np
from tqdm import tqdm


def velocity_verlet(
    mass, pos, vel, forces, particles, dt, steps, output_filename="vel_verlet.npy"
):
    """_summary_

    Args:
        mass (_type_): _description_
        pos (_type_): _description_
        vel (_type_): _description_
        forces (_type_): _description_
        particles (_type_): _description_
        dt (_type_): timesteps in half a day
        steps (_type_): max steps
        output_filename (str, optional): _description_. Defaults to "positions.npy".
    """
    # init forces
    forces = system_forces(mass, pos)
    next_step_forces = np.zeros((particles, 3))

    # data storage
    positions_over_time = np.zeros((steps, particles, 3))
    velocities_over_time = np.zeros((steps, particles, 3))
    positions_over_time[0] = pos
    velocities_over_time[0] = vel

    for step in tqdm(range(1, steps), desc="Velocity Verlet Progress"):
        # update positions
        new_pos = pos + dt * vel + dt**2 / (2 * mass) * forces
        # calculate force for the next step with new positions
        next_step_forces = system_forces(mass, new_pos)
        new_vel = vel + dt / (2 * mass) * (forces + next_step_forces)
        positions_over_time[step] = new_pos
        velocities_over_time[step] = new_vel
        forces = next_step_forces
        pos = new_pos
        vel = new_vel

    basis_pfad = os.path.dirname(os.path.dirname(__file__))
    filename = os.path.join(basis_pfad, "output", "positions_" + output_filename)
    np.save(filename, positions_over_time)
    filename = os.path.join(basis_pfad, "output", "velocities_" + output_filename)
    np.save(filename, velocities_over_time)


def verlet(
    mass,
    pos0,
    pos1,
    vel0,
    forces,
    particles,
    dt,
    steps,
    output_filename="verlet.npy",
):
    """_summary_

    Args:
        mass (_type_): _description_
        pos0 (_type_): _description_
        pos1 (_type_): _description_
        forces (_type_): _description_
        particles (_type_): _description_
        dt (_type_): _description_
        steps (_type_): _description_
        output_filename (str, optional): _description_. Defaults to "positions_verlet.npy".
    """
    positions_over_time = np.zeros((steps, particles, 3))
    velocities_over_time = np.zeros((steps, particles, 3))
    positions_over_time[0] = pos0
    positions_over_time[1] = pos1
    velocities_over_time[0] = vel0
    for step in tqdm(range(1, steps - 1), desc="Verlet Progress"):
        forces = system_forces(mass, pos0)  # f(0)
        pos2 = 2 * pos1 - pos0 + dt / mass * forces
        vel = (pos2 - pos0) / (2 * dt)
        velocities_over_time[step] = vel
        positions_over_time[step + 1] = pos2
        pos0 = pos1
        pos1 = pos2
    basis_pfad = os.path.dirname(os.path.dirname(__file__))
    filename = os.path.join(basis_pfad, "output", "positions_" + output_filename)
    np.save(filename, positions_over_time)
    filename = os.path.join(basis_pfad, "output", "velocities_" + output_filename)
    np.save(filename, velocities_over_time)


def euler(mass, pos, vel, forces, particles, dt, steps, output_filename="euler.npy"):
    """_summary_

    Args:
        mass (_type_): _description_
        pos (_type_): _description_
        vel (_type_): _description_
        forces (_type_): _description_
        particles (_type_): _description_
        dt (_type_): timesteps in half a day
        steps (_type_): max steps
        output_filename (str, optional): _description_. Defaults to "positions_euler.npy".
    """
    # init forces
    forces = system_forces(mass, pos)
    next_step_forces = np.zeros((particles, 3))

    # data storage
    positions_over_time = np.zeros((steps, particles, 3))
    velocities_over_time = np.zeros((steps, particles, 3))
    positions_over_time[0] = pos
    velocities_over_time[0] = vel

    for step in tqdm(range(1, steps), desc="Euler Progress"):
        # update positions
        new_pos = pos + dt * vel + dt**2 / (2 * mass) * forces
        # calculate force for the next step with new positions
        next_step_forces = system_forces(mass, new_pos)
        new_vel = vel + dt / (2 * mass) * (forces)
        positions_over_time[step] = new_pos
        velocities_over_time[step] = new_vel
        forces = next_step_forces
        pos = new_pos
        vel = new_vel

    basis_pfad = os.path.dirname(os.path.dirname(__file__))
    filename = os.path.join(basis_pfad, "output", "positions_" + output_filename)
    np.save(filename, positions_over_time)
    filename = os.path.join(basis_pfad, "output", "velocities_" + output_filename)
    np.save(filename, velocities_over_time)
