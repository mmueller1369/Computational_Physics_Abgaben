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
    # before the run
    ## initialize the files and positions etc
    files = create_files(trajfile, tempfile, energyfile, rdffile)
    trajfile, tempfile, energyfile, rdffile = files
    x, y, z, vx, vy, vz, fx, fy, fz = initial_config
    ## get the force and thermostat function
    force_func = getattr(forces, f"force{force}")
    if thermostat:
        thermostat_func = getattr(tools, thermostat)
    ## initialize the histogram if needed
    if rdffile:
        n_bins = settings.rmax_hist//settings.dr_hist
        n_gr = steps//n_save
        histograms = np.zeros(shape=(n_gr, n_bins))
    else:
        histograms = None
    ## calculate the initial forces and energies
    fx, fy, fz, energies = force_func(x, y, z, *force_params)

    # conduct the run
    for step in tqdm(prange(0, steps), desc=simulation_name):
        ## save the specified parameters of the current state
        if step % n_save == 0:
            save_specified_properties(trajfile, tempfile, energyfile, rdffile,
                step, x, y, z, vx, vy, vz, fx, fy, fz, energies, n_save, histograms)
        ## integrate equations of motion
        x, y, z, vx, vy, vz, fx, fy, fz, energies = integrators.VelocityVerlet(
            x, y, z,
            vx, vy, vz,
            fx, fy, fz,
            settings.deltat,
            *force_params)
        ## apply the thermostat
        if thermostat and step % n_thermostat == 0:
            Tnow = tools.computeTemperature(vx, vy, vz, settings.masses)
            vx, vy, vz = thermostat_func(vx, vy, vz, Tnow, *thermostat_params)
            
    # after the run
    ## calculate the final rdf function from all the histograms
    if rdffile:
        rdf, _ = histogram.calc_RDF(histograms)
        r = np.arange(0, len(rdf)) * settings.dr_hist
        for ri, gi in zip(r, rdf):
            rdffile.write("%e %e\n" % (ri, gi))
    ## close all files
    for file in files:
        if file:
            file.close()
    ## return the final positions so that the simulation can be continued (e.g. after equilibration)
    return x, y, z, vx, vy, vz, fx, fy, fz


def create_files(trajfile, tempfile, energyfile, rdffile):
    if trajfile:
        trajfile = open(os.path.join(settings.path, f"{trajfile}.txt"), "w")
    if tempfile:
        tempfile = open(os.path.join(settings.path, f"{tempfile}.txt"), "w")
        tempfile.write("# step T\n")
    if energyfile:
        energyfile = open(os.path.join(settings.path, f"{energyfile}.txt"), "w")
        energyfile.write("# step e_LJ e_coul e_bond e_angle e_kin\n")
    if rdffile:
        rdffile = open(os.path.join(settings.path, f"{rdffile}.txt"), "w")
        rdffile.write("# r g(r)\n")
    return (trajfile, tempfile, energyfile, rdffile)


def save_specified_properties(
    trajfile, tempfile, energyfile, rdffile, # files to be written in; None if parameter is skipped
    step, x, y, z, vx, vy, vz, fx, fy, fz, energies, # current properties of the system
    n_save, histograms # parameters needed for the histogramming
):
    if trajfile:
        export.WriteTrajectory(trajfile, step, x, y, z, vx, vy, vz, fx, fy, fz)
    if tempfile:
        export.WriteTemperature(tempfile, step, vx, vy, vz)
    if energyfile:
        export.WriteEnergy(energyfile, step, vx, vy, vz, *energies)
    if rdffile:
        t = step//n_save
        histograms[t] = histogram.create_histogram(x, y, z)