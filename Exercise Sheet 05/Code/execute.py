import os
import settings
import misc
import forces
import update
import initialize
import g_r
import numpy as np
from tqdm import tqdm
from numba import prange

settings.init()


def create_files(trajfile, tempfile, energyfile, rdffile):
    if trajfile:
        trajfile = open(os.path.join(settings.path, f"{trajfile}.txt"), "w")
    if tempfile:
        tempfile = open(os.path.join(settings.path, f"{tempfile}.txt"), "w")
        tempfile.write("#step  T\n")
    if energyfile:
        energyfile = open(os.path.join(settings.path, f"{energyfile}.txt"), "w")
        energyfile.write("#step  PE  KE  vx2 vy2 vz2\n")
    if rdffile:
        rdffile = open(os.path.join(settings.path, f"{rdffile}.txt"), "w")
        rdffile.write("#r/sigma g(r)\n")
    return trajfile, tempfile, energyfile, rdffile


def run_simulation(
    initial_config,
    integrator,
    force,
    steps,
    thermostat=False,
    thermostat_params=False,
    n_thermostat=1,
    trajfile=False,
    tempfile=False,
    energyfile=False,
    rdffile=False,
    n_save=settings.n_save,
    simulation_name="Simulation",
):
    # initializing everything
    trajfile, tempfile, energyfile, rdffile = create_files(
        trajfile, tempfile, energyfile, rdffile
    )
    x, y, z, vx, vy, vz, fx, fy, fz = initial_config
    force_func = getattr(forces, f"force{force}")
    force_params = getattr(misc, f"params{force}")()
    integrator_func = getattr(update, integrator)
    if thermostat:
        thermostat_func = getattr(initialize, thermostat)
    if rdffile:
        histogram, bin_width = initialize.histogram()

    # write first trajectory
    if trajfile:
        misc.WriteTrajectory(trajfile, 0, x, y, z, vx, vy, vz, fx, fy, fz)
    # calculate initial forces
    fx, fy, fz, epot = force_func(x, y, z, *force_params)

    # start the run
    for step in tqdm(prange(0, steps), desc=simulation_name):
        x, y, z, vx, vy, vz, fx, fy, fz, epot = integrator_func(
            x,
            y,
            z,
            vx,
            vy,
            vz,
            fx,
            fy,
            fz,
            *force_params,
            settings.deltat,
            settings.mass,
        )

        if thermostat and step % n_thermostat == 0:
            Tsystem = initialize.temperature(vx, vy, vz)
            vx, vy, vz = thermostat_func(
                vx, vy, vz, thermostat_params[0], Tsystem, *thermostat_params[1:]
            )

        if step % n_save == 0:
            if trajfile:
                misc.WriteTrajectory(trajfile, step, x, y, z, vx, vy, vz, fx, fy, fz)
            if tempfile:
                misc.WriteTemp(tempfile, step, vx, vy, vz)
            if energyfile:
                ekin = update.KineticEnergy(vx, vy, vz, settings.mass)
                vx2, vy2, vz2 = misc.squarevelocity(vx, vy, vz, settings.mass)
                misc.WriteEnergy(energyfile, step, epot, ekin, vx2, vy2, vz2)
            if rdffile:
                t = int(step / settings.n_analyze)
                histogram[t] = g_r.histogram(x, y, z, bin_width, settings.rmax)

    if rdffile:
        rdf, _ = g_r.calc_RDF(histogram, bin_width)
        r = np.arange(0, len(rdf)) * bin_width / settings.sigma
        for ri, gi in zip(r, rdf):
            rdffile.write("%e %e\n" % (ri, gi))

    for file in [trajfile, tempfile, energyfile]:
        if file:
            file.close()

    return x, y, z, vx, vy, vz, fx, fy, fz
