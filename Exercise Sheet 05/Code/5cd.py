# simulation and analysis of part a

import settings
import initialize
import force
import update
import time
import misc
import numpy as np
from tqdm import tqdm
import os
import force
from numba import prange


start = time.time()

# initialization of global variable
settings.init()
fileoutput = open(os.path.join(settings.path, f"trajectories_c_eq.txt"), "w")
filetemp = open(os.path.join(settings.path, f"temp_c_eq.txt"), "w")
filetemp.write("#step  T\n")

# create atomic locations and velocities + cancel linear momentum + rescale velocity to desired temperature
x, y, z, vx, vy, vz = initialize.InitializeAtoms()
f_initial = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
# save configuration to visualize
misc.WriteTrajectory(
    fileoutput, 0, x, y, z, vx, vy, vz, f_initial, f_initial, f_initial
)

# initialize the forces
(xlo, xhi, ylo, yhi, zlo, zhi, eps, sigma, cutoff, deltat, mass) = misc.inputset()
fx, fy, fz, epot = force.forceLJ(
    x, y, z, xlo, xhi, ylo, yhi, zlo, zhi, eps, sigma, cutoff
)
eps = 0.5 * settings.kb * settings.Tdesired

# -------------- Equilibration ---------------#
for step in tqdm(prange(0, settings.nsteps_equi), desc="Equilibration"):

    x, y, z, vx, vy, vz, fx, fy, fz, epot = update.VelocityVerlet(
        x,
        y,
        z,
        vx,
        vy,
        vz,
        fx,
        fy,
        fz,
        xlo,
        xhi,
        ylo,
        yhi,
        zlo,
        zhi,
        eps,
        sigma,
        cutoff,
        deltat,
        mass,
    )

    if (
        settings.thermostat == 1 and step % settings.n_thermostat == 0
    ):  # rescaling of the temperature # the following lines should be defined as a routine in misc
        Trandom = initialize.temperature(vx, vy, vz)
        vx, vy, vz = initialize.rescalevelocity(vx, vy, vz, settings.Tdesired, Trandom)
        Trandom1 = initialize.temperature(vx, vy, vz)

    if (
        settings.thermostat == 2
    ):  # rescaling of the temperature # the following lines should be defined as a routine in misc
        Trandom = initialize.temperature(vx, vy, vz)
        vx, vy, vz = initialize.berendsen_thermostat(
            vx, vy, vz, settings.Tdesired, Trandom, settings.tau, settings.deltat
        )
        Trandom1 = initialize.temperature(vx, vy, vz)

    if step % settings.n_save == 0:  # save the trajectory
        ekin = update.KineticEnergy(vx, vy, vz, mass)  # calculate the kinetic energy
        misc.WriteTemp(filetemp, step, vx, vy, vz)
        misc.WriteTrajectory(fileoutput, step, x, y, z, vx, vy, vz, fx, fy, fz)

    force.forceLJ(
        x,
        y,
        z,
        xlo,
        xhi,
        ylo,
        yhi,
        zlo,
        zhi,
        eps,
        sigma,
        cutoff,
    )

fileoutput.close()
filetemp.close()


# -------------- Production ---------------#
fileoutput = open(os.path.join(settings.path, f"trajectories_c_prod.txt"), "w")
filetemp = open(os.path.join(settings.path, f"temp_c_prod.txt"), "w")
filetemp.write("#step  T\n")
for step in tqdm(prange(0, settings.nsteps_production), desc="Production"):

    x, y, z, vx, vy, vz, fx, fy, fz, epot = update.VelocityVerlet(
        x,
        y,
        z,
        vx,
        vy,
        vz,
        fx,
        fy,
        fz,
        xlo,
        xhi,
        ylo,
        yhi,
        zlo,
        zhi,
        eps,
        sigma,
        cutoff,
        deltat,
        mass,
    )

    if (
        settings.thermostat == 1 and step % settings.n_thermostat == 0
    ):  # rescaling of the temperature # the following lines should be defined as a routine in misc
        Trandom = initialize.temperature(vx, vy, vz)
        vx, vy, vz = initialize.rescalevelocity(vx, vy, vz, settings.Tdesired, Trandom)
        Trandom1 = initialize.temperature(vx, vy, vz)

    if (
        settings.thermostat == 2
    ):  # rescaling of the temperature # the following lines should be defined as a routine in misc
        Trandom = initialize.temperature(vx, vy, vz)
        vx, vy, vz = initialize.berendsen_thermostat(
            vx, vy, vz, settings.Tdesired, Trandom, settings.tau, settings.deltat
        )
        Trandom1 = initialize.temperature(vx, vy, vz)

    if step % settings.n_save == 0:  # save the trajectory
        ekin = update.KineticEnergy(vx, vy, vz, mass)  # calculate the kinetic energy
        misc.WriteTemp(filetemp, step, vx, vy, vz)
        misc.WriteTrajectory(fileoutput, step, x, y, z, vx, vy, vz, fx, fy, fz)

    force.forceLJ(
        x,
        y,
        z,
        xlo,
        xhi,
        ylo,
        yhi,
        zlo,
        zhi,
        eps,
        sigma,
        cutoff,
    )

fileoutput.close()
filetemp.close()

print("total time = ", time.time() - start)
