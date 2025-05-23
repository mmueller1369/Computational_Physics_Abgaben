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
import g_r
from tqdm import tqdm
import os

start = time.time()

# initialization of global variable
settings.init()
fileoutput = open(os.path.join(settings.path, "trajectories_eq"), "w")
fileenergy = open(os.path.join(settings.path, "energies_eq"), "w")
fileenergy.write("#step  PE  KE  vx2 vy2 vz2\n")

# create atomic locations and velocities + cancel linear momentum + rescale velocity to desired temperature
x, y, z, vx, vy, vz = initialize.InitializeAtoms()
f_initial = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
# save configuration to visualize
misc.WriteTrajectory(
    fileoutput, 0, x, y, z, vx, vy, vz, f_initial, f_initial, f_initial
)

# initialize the forces
(
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
    eps_wall,
    sigma_wall,
    cutoff_wall,
) = misc.inputset()
fx, fy, fz, epot = force.forceLJ(
    x, y, z, xlo, xhi, ylo, yhi, zlo, zhi, eps, sigma, cutoff
)

# -------------- EQUILIBRATION ---------------#
for step in tqdm(range(0, settings.nsteps_equi), desc="Equalibration"):

    x, y, z, vx, vy, vz, fx, fy, fz, epot = update.VelocityVerlet_wall_z(
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
        eps_wall,
        sigma_wall,
        cutoff_wall,
    )

    if (
        settings.Trescale == 1 and step % 10 == 0
    ):  # rescaling of the temperature # the following lines should be defined as a routine in misc
        Trandom = initialize.temperature(vx, vy, vz)
        vx, vy, vz = initialize.rescalevelocity(vx, vy, vz, settings.Tdesired, Trandom)
        Trandom1 = initialize.temperature(vx, vy, vz)

    if step % 10 == 0:  # save the trajectory
        ekin = update.KineticEnergy(vx, vy, vz, mass)  # calculate the kinetic energy
        vx2, vy2, vz2 = misc.squarevelocity(
            vx, vy, vz, mass
        )  # calculate v_x^2 to compare with 0.5Nk_BT
        misc.WriteEnergy(fileenergy, step, epot, ekin, vx2, vy2, vz2)
        misc.WriteTrajectory(fileoutput, step, x, y, z, vx, vy, vz, fx, fy, fz)

fileoutput.close()
fileenergy.close()

# -------------- PRODUCTION ---------------#
fileoutput = open(os.path.join(settings.path, "trajectories_prod"), "w")
fileenergy = open(os.path.join(settings.path, "energies_prod"), "w")
fileenergy.write("#step  PE  KE  vx2 vy2 vz2\n")
settings.Trescale = 0
histogram_x, bin_width = initialize.histogram_1d(xhi, xlo)
histogram_y, bin_width = initialize.histogram_1d(yhi, ylo)
histogram_z, bin_width = initialize.histogram_1d(zhi, zlo)

density_profiles, bin_width = initialize.block_histograms(zhi, zlo)

for step in tqdm(range(0, settings.nsteps_production), desc="Production"):

    x, y, z, vx, vy, vz, fx, fy, fz, epot = update.VelocityVerlet_wall_z(
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
        eps_wall,
        sigma_wall,
        cutoff_wall,
    )

    if step % 10 == 0:  # save the trajectory
        misc.WriteTrajectory(fileoutput, step, x, y, z, vx, vy, vz, fx, fy, fz)
        ekin = update.KineticEnergy(vx, vy, vz, mass)  # calculate the kinetic energy
        vx2, vy2, vz2 = misc.squarevelocity(
            vx, vy, vz, mass
        )  # calculate v_x^2 to compare with 0.5Nk_BT
        misc.WriteEnergy(fileenergy, step, epot, ekin, vx2, vy2, vz2)

    # calculate the radial distribution function
    if step % settings.n_analyze == 0:
        t = int(step / settings.n_analyze)
        histogram_x[t] = g_r.histogram_1d(x, xlo, xhi)
        histogram_y[t] = g_r.histogram_1d(y, ylo, yhi)
        histogram_z[t] = g_r.histogram_1d(z, zlo, zhi)

    if step % settings.block_length == 0:
        t = int(step / settings.block_length)
        hist = g_r.histogram_1d(z, zlo, zhi)
        density_profiles[t] = g_r.calc_density_1d(hist)


fileoutput.close()
fileenergy.close()

np.savetxt(os.path.join(settings.path, "histogram_x.txt"), histogram_x)
np.savetxt(os.path.join(settings.path, "histogram_y.txt"), histogram_y)
np.savetxt(os.path.join(settings.path, "histogram_z.txt"), histogram_z)
# save density profiles for each block
np.savetxt(os.path.join(settings.path, "density_profiles.txt"), density_profiles)

print("total time = ", time.time() - start)
