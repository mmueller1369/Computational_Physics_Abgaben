# settings

# velocity: m*s^-1
# position: m
# acceleration: ms^-2
# energy: joule
# temperature: K

import numpy as np


def init():

    global nsteps  # number of time step to analyze
    nsteps = 10000
    global mass  # mass of the LJ particles (gram/mole)
    mass = 39.95
    global kb  # boltzmann's constant (kcal/mole/K)
    kb = 0.0019858775
    global Tdesired  # temperature of the experiment in K
    Tdesired = 300.0
    global eps  # eps in LJ (kcal/mole)
    eps = 0.29788162
    global r0  # r0 in LJ (nm)
    r0 = 0.188
    global cutoff  # cutoff arbitrary at 2.5 r0
    cutoff = 2.5 * r0
    global deltat  # time step (fs)
    deltat = 1

    # number of particle = n1*n2 distributed on s square lattice
    global n1
    n1 = 5
    global n2
    n2 = 5
    global n3
    n3 = 5

    # desired density
    rho = 0.01  # N/V = 0.01 sigma^-3
    l = (n1 * n2 * n3 / rho) ** (1 / 3)  # l = (N/V)^(1/3) = (n1*n2*n3/rho)^(1/3)

    # box size
    global xlo
    xlo = 0 * r0
    global xhi
    xhi = l * r0
    global ylo
    ylo = 0 * r0
    global yhi
    yhi = l * r0
    global zlo
    zlo = 0 * r0
    global zhi
    zhi = l * r0

    global deltaxyz  # lattice parameter to setup the initial configuration on a lattice
    deltaxyz = (xhi - xlo) / n1

    # rescaling of temperature
    global Trescale
    Trescale = 1  # 1 = rescale temperature; 0 = no rescaling
