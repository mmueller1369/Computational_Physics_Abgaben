import os
import numpy as np
from tqdm import tqdm
from numba import prange
import forces
import tools
import integrators
import export
import settings
# import settings_SI as settings
settings.init()


def run_simulation(
    initial_config,
    force,
    force_params,
    steps,
    masses=settings.masses,
    thermostat=False,
    thermostat_params=False,
    n_thermostat=1,
    trajfile=False,
    tempfile=False,
    energyfile=False,
    n_save=settings.n_save,
    simulation_name="Simulation",
):
    # before the run
    ## initialize the files and positions etc
    files = create_files(trajfile, tempfile, energyfile)
    trajfile, tempfile, energyfile = files
    x, y, z, vx, vy, vz, fx, fy, fz = initial_config
    ## get the force and thermostat function
    force_func = getattr(forces, f"force{force}")
    if thermostat:
        thermostat_func = getattr(tools, f"{thermostat}Thermostat")
    ## calculate the initial forces and energies
    fx, fy, fz, energies = force_func(x, y, z, *force_params)

    # conduct the run
    for step in tqdm(prange(0, steps), desc=simulation_name):
        ## save the specified parameters of the current state
        if step % n_save == 0:
            save_specified_properties(trajfile, tempfile, energyfile,
                step, x, y, z, vx, vy, vz, fx, fy, fz, energies, masses)
        ## integrate equations of motion
        x, y, z, vx, vy, vz, fx, fy, fz, energies = integrators.VelocityVerlet(
            x, y, z,
            vx, vy, vz,
            fx, fy, fz,
            masses,
            settings.deltat,
            force, force_params)
        ## apply the thermostat
        if thermostat and step % n_thermostat == 0:
            Tnow = tools.computeTemperature(vx, vy, vz, masses)
            vx, vy, vz = thermostat_func(vx, vy, vz, Tnow, *thermostat_params)
            
    # after the run
    ## close all files
    for file in files:
        if file:
            file.close()
    ## return the final positions so that the simulation can be continued (e.g. after equilibration)
    return x, y, z, vx, vy, vz, fx, fy, fz


def create_files(trajfile, tempfile, energyfile):
    if trajfile:
        trajfile = open(os.path.join(settings.path, f"{trajfile}.txt"), "w")
    if tempfile:
        tempfile = open(os.path.join(settings.path, f"{tempfile}.txt"), "w")
        tempfile.write("# time T\n")
    if energyfile:
        energyfile = open(os.path.join(settings.path, f"{energyfile}.txt"), "w")
        energyfile.write("# time e_LJ e_coul e_bond e_angle e_kin\n")
    return (trajfile, tempfile, energyfile)


def save_specified_properties(
    trajfile, tempfile, energyfile, # files to be written in; None if parameter is skipped
    step, x, y, z, vx, vy, vz, fx, fy, fz, energies, masses, # current properties of the system
):
    if trajfile:
        export.WriteTrajectory(trajfile, step, x, y, z, vx, vy, vz, fx, fy, fz, masses)
    if tempfile:
        export.WriteTemperature(tempfile, step, vx, vy, vz, masses)
    if energyfile:
        export.WriteEnergy(energyfile, step, vx, vy, vz, *energies, masses)