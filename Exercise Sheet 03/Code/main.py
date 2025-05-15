# MD code in microcanonical ensemble
# force field = LJ with eps = alpha * kb * T and sigma = 15 angstrom
# the treatment of the discontinuity at the cutoff is not implemented, i.e. the potential does not go smoothly to 0 at r = cutoff

import settings
import initialize
import force
import update
import debug
import sys
import time
import misc
import numpy as np
from tqdm import tqdm

start = time.time()


fileoutput = open("output_equilibration.txt", "w")
fileenergy = open("energy_equilibration.txt", "w")
fileenergy.write("#step  PE  KE  vx2 vy2\n")
# initialization of global variable
settings.init()

# create atomic locations and velocities + cancel linear momentum + rescale velocity to desired temperature
x, y, z, vx, vy, vz = initialize.InitializeAtoms()
f_initial = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
# save configuration to visualize
misc.WriteTrajectory(
    fileoutput, 0, x, y, z, vx, vy, vz, f_initial, f_initial, f_initial
)

# initialize the forces
xlo, xhi, ylo, yhi, zlo, zhi, eps, sigma, cutoff, deltat, mass = misc.inputset()
fx, fy, fz, epot = force.forceLJ(
    x, y, z, xlo, xhi, ylo, yhi, zlo, zhi, eps, sigma, cutoff
)

# -------------- EQUILIBRATION ---------------#
for step in tqdm(range(0, 1 * settings.nsteps)):  # equilibration

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
        settings.Trescale == 1 and step % 10 == 0
    ):  # rescaling of the temperature # the following lines should be defined as a routine in misc
        Trandom = initialize.temperature(vx, vy, vz)
        vx, vy, vz = initialize.rescalevelocity(vx, vy, vz, settings.Tdesired, Trandom)
        Trandom1 = initialize.temperature(vx, vy, vz)

    if step % 100 == 0:  # save the trajectory
        ekin = update.KineticEnergy(vx, vy, vz, mass)  # calculate the kinetic energy
        vx2, vy2, vz2 = misc.squarevelocity(
            vx, vy, vz, mass
        )  # calculate v_x^2 to compare with 0.5Nk_BT
        misc.WriteEnergy(fileenergy, step, epot, ekin, vx2, vy2, vz2)
        misc.WriteTrajectory(fileoutput, step, x, y, z, vx, vy, vz, fx, fy, fz)

fileoutput.close()
fileenergy.close()

# -------------- PRODUCTION ---------------#
fileoutput = open("output_prod.txt", "w")
fileenergy = open("energy_prod.txt", "w")
fileenergy.write("#step  PE  KE  vx2 vy2 vz2\n")
settings.Trescale = 0
for step in tqdm(range(0, 2 * settings.nsteps)):  # production

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

    if step % 100 == 0:  # save the trajectory
        misc.WriteTrajectory(fileoutput, step, x, y, z, vx, vy, vz, fx, fy, fz)
        ekin = update.KineticEnergy(vx, vy, vz, mass)  # calculate the kinetic energy
        vx2, vy2, vz2 = misc.squarevelocity(
            vx, vy, vz, mass
        )  # calculate v_x^2 to compare with 0.5Nk_BT
        misc.WriteEnergy(fileenergy, step, epot, ekin, vx2, vy2, vz2)

fileoutput.close()
fileenergy.close()

print("total time = ", time.time() - start)
