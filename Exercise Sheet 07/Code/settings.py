# settings

# velocity: m*s^-1
# position: m
# acceleration: ms^-2
# energy: joule
# temperature: K

import numpy as np
import os


def init():

    global nsteps_equi  # number of time step to analyze
    nsteps_equi = 5000
    global nsteps_production
    nsteps_production = 495000
    global mass  # mass of the LJ particles (gram/mole)
    mass = 13.97
    global kb  # boltzmann's constant (kcal/mole/K)
    kb = 0.0019849421
    global Tdesired  # temperature of the experiment in K
    Tdesired = 300.0
    global eps  # eps in LJ (kcal/mole)
    eps = 0.0595
    global sigma  # sigma in LJ (nm)
    sigma = 0.326
    global deltat  # time step (fs)
    deltat = 1
    # diatomic stuff
    global b0
    b0 = 0.107
    global kb_di
    kb_di = 9793

    # number of particle = n1*n2 distributed on s square lattice
    global n1
    n1 = 5
    global n2
    n2 = 5
    global n3
    n3 = 5
    global nparticles
    nparticles = n1 * n2 * n3

    # desired density
    global rho
    rho = 0.25  # N/V = 0.01 sigma^-3

    # box lengths in each direction
    global lx
    lx = n1 / (rho ** (1 / 3))
    global ly
    ly = n2 / (rho ** (1 / 3))
    global lz
    lz = n3 / (rho ** (1 / 3))

    # box size
    global xlo
    xlo = 0 * sigma
    global xhi
    xhi = lx * sigma
    global ylo
    ylo = 0 * sigma
    global yhi
    yhi = ly * sigma
    global zlo
    zlo = 0 * sigma
    global zhi
    zhi = lz * sigma

    global deltaxyz  # lattice parameter to setup the initial configuration on a lattice
    deltaxyz = (xhi - xlo) / n1

    # rescaling of temperature
    global thermostat
    thermostat = (
        2  # 1 = rescale temperature; 0 = no rescaling, 2 = berendsen, 3 = andersen
    )
    global tau
    tau = 500 * deltat
    global n_thermostat
    n_thermostat = 1
    global nu  # collision frequency for Andersen thermostat
    nu = 0.01  # in fs^-1

    global deltar  # bin size for histogram
    deltar = 0.05 * sigma
    global rmax  # maximum distance for histogram
    rmax = 1 / 2 * max(lx, ly, lz) * sigma  # should be 1/2 of the box size
    global n_analyze  # every n_analyze steps, the histogram is calculated
    n_analyze = 10
    global n_gr
    n_gr = int(nsteps_production / n_analyze)

    # number of blocks
    global nblocks
    nblocks = 6

    global path  # path to the output files
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    global n_save
    n_save = 10

