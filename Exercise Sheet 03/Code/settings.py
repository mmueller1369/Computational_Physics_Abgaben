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
    nsteps_equi = 10000
    global nsteps_production
    nsteps_production = 20000
    global mass  # mass of the LJ particles (gram/mole)
    mass = 39.95
    global kb  # boltzmann's constant (kcal/mole/K)
    kb = 0.0019858775
    global Tdesired  # temperature of the experiment in K
    Tdesired = 300.0
    global eps  # eps in LJ (kcal/mole)
    eps = 0.29788162
    global sigma  # sigma in LJ (nm)
    sigma = 0.188
    global cutoff  # cutoff arbitrary at 2.5 sigma
    cutoff = 2.5 * sigma
    global deltat  # time step (fs)
    deltat = 1

    # number of particle = n1*n2 distributed on s square lattice
    global n1
    n1 = 8
    global n2
    n2 = 8
    global n3
    n3 = 8

    # desired density
    global rho
    rho = 0.01  # N/V = 0.01 sigma^-3
    l = (n1 * n2 * n3 / rho) ** (1 / 3)  # l = (N/V)^(1/3) = (n1*n2*n3/rho)^(1/3)

    # box size
    global xlo
    xlo = 0 * sigma
    global xhi
    xhi = l * sigma
    global ylo
    ylo = 0 * sigma
    global yhi
    yhi = l * sigma
    global zlo
    zlo = 0 * sigma
    global zhi
    zhi = l * sigma

    global deltaxyz  # lattice parameter to setup the initial configuration on a lattice
    deltaxyz = (xhi - xlo) / n1

    # rescaling of temperature
    global Trescale
    Trescale = 1  # 1 = rescale temperature; 0 = no rescaling

    global deltar  # bin size for histogram
    deltar = 0.01 * sigma
    global rmax  # maximum distance for histogram
    rmax = 1 / 2 * l * sigma  # should be 1/2 of the box size
    global n_analyze  # every n_analyze steps, the histogram is calculated
    n_analyze = 10
    global n_gr
    n_gr = int(nsteps_production / n_analyze)

    global path  # path to the output files
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
