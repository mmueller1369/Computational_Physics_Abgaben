import os
import settings
import misc
import forces
import update
import initialize
from tqdm import tqdm
from numba import prange

settings.init()


def create_files(trajfile, tempfile, energyfile):
    if trajfile:
        trajfile = open(os.path.join(settings.path, f"{trajfile}.txt"), "w")
    if tempfile:
        tempfile = open(os.path.join(settings.path, f"{tempfile}.txt"), "w")
        tempfile.write("#step  T\n")
    if energyfile:
        energyfile = open(os.path.join(settings.path, f"{energyfile}.txt"), "w")
        energyfile.write("#step  PE  KE  vx2 vy2 vz2\n")
    return trajfile, tempfile, energyfile


def run_simulation(
    initial_config,
    integrator,
    force,
    force_params,
    steps,
    thermostat=False,
    thermostat_params=False,
    n_thermostat=1,
    trajfile=False,
    tempfile=False,
    energyfile=False,
    n_save=settings.n_save,
    simulation_name="Simulation",
):
    # initializing everything
    trajfile, tempfile, energyfile = create_files(trajfile, tempfile, energyfile)
    x, y, z, vx, vy, vz, fx, fy, fz = initial_config
    force_func = getattr(forces, force)
    integrator_func = getattr(update, integrator)
    if thermostat:
        thermostat_func = getattr(initialize, thermostat)
    # write first trajectory
    if trajfile:
        misc.WriteTrajectory(trajfile, 0, x, y, z, vx, vy, vz, fx, fy, fz)
    # calculate initial forces
    fx, fy, fz, epot = force_func(x, y, z, *force_params)

    # start the run
    for step in tqdm(prange(0, steps), desc=simulation_name):
        x, y, z, vx, vy, vz, fx, fy, fz, epot = integrator_func(
            x, y, z, vx, vy, vz, fx, fy, fz, *force_params
        )

    if thermostat and step % n_thermostat == 0:
        Tsystem = initialize.temperature(vx, vy, vz)
        vx, vy, vz = thermostat_func(
            vx, vy, vz, thermostat_params[0], Tsystem, *thermostat_params[1:]
        )

    if step % n_save == 0:  # save the trajectory
        if trajfile:
            misc.WriteTrajectory(trajfile, step, x, y, z, vx, vy, vz, fx, fy, fz)
        if tempfile:
            misc.WriteTemp(tempfile, step, vx, vy, vz)
        if energyfile:
            ekin = update.KineticEnergy(vx, vy, vz, settings.mass)


fileoutput.close()
filetemp.close()
