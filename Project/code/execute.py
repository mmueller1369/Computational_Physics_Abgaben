import os
import numpy as np
from tqdm import tqdm
from numba import prange
import forces
import tools
import integrators
import histogram
import export
import settings
settings.init()


def create_files(trajfile, tempfile, energyfile, rdffile):
    if trajfile:
        trajfile = open(os.path.join(settings.path, f"{trajfile}.txt"), "w")
    if tempfile:
        tempfile = open(os.path.join(settings.path, f"{tempfile}.txt"), "w")
        tempfile.write("#step  T\n")
    if energyfile:
        energyfile = open(os.path.join(settings.path, f"{energyfile}.txt"), "w")
        energyfile.write("#step  e_LJ  e_coul  e_bond  e_angle  e_kin")
    if rdffile:
        rdffile = open(os.path.join(settings.path, f"{rdffile}.txt"), "w")
        rdffile.write("#r  g(r)\n")
    return (trajfile, tempfile, energyfile, rdffile)

#force params, thermostat params (Tdesired, tau, dt) in main

def run_simulation(
    initial_config,
    force,
    force_params,
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
    files = create_files(trajfile, tempfile, energyfile, rdffile)
    trajfile, tempfile, energyfile, rdffile = files
    x, y, z, vx, vy, vz, fx, fy, fz = initial_config
    force_func = getattr(forces, f"force{force}")
    if thermostat:
        thermostat_func = getattr(tools, thermostat)
    if rdffile:
        n_bins = settings.rmax_hist//settings.dr_hist
        n_gr = steps//n_save
        histograms = np.zeros(shape=(n_gr, n_bins))

    # write first trajectory and calculate initial forces
    if trajfile:
        export.WriteTrajectory(trajfile, 0, x, y, z, vx, vy, vz, fx, fy, fz)
    fx, fy, fz, epot = force_func(x, y, z, *force_params)

    # start the run
    for step in tqdm(prange(0, steps), desc=simulation_name):
        # integrate equations of motion
        x, y, z, vx, vy, vz, fx, fy, fz, energies = integrators.VelocityVerlet(
            x, y, z,
            vx, vy, vz,
            fx, fy, fz,
            settings.deltat,
            *force_params)

        # apply the thermostat
        if thermostat and step % n_thermostat == 0:
            Tnow = tools.computeTemperature(vx, vy, vz, settings.masses)
            vx, vy, vz = thermostat_func(vx, vy, vz, vx, vy, vz, Tnow, *thermostat_params)

        # save the specified parameters
        if step % n_save == 0:
            if trajfile:
                export.WriteTrajectory(trajfile, step, x, y, z, vx, vy, vz, fx, fy, fz)
            if tempfile:
                export.WriteTemperature(tempfile, step, vx, vy, vz)
            if energyfile:
                export.WriteEnergy(energyfile, step, vx, vy, vz, *energies)
            if rdffile:
                t = step//n_save
                histograms[t] = histogram.create_histogram(x, y, z)

    if rdffile:
        rdf, _ = histogram.calc_RDF(histograms)
        r = np.arange(0, len(rdf)) * settings.dr_hist
        for ri, gi in zip(r, rdf):
            rdffile.write("%e %e\n" % (ri, gi))

    for file in files:
        if file:
            file.close()

    return x, y, z, vx, vy, vz, fx, fy, fz