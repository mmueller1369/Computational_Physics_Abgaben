"""
--------- units ---------
length: nm
time: fs
mass: gram/mole
force: (kcal/mole)/nm
temperature: K
velocity: nm/fs
angle: rad
charge: elementary charge e



[epsilon] = kcal/mole;
[k_B] = (kcal/mole)/K;
[theta]
conversion factor:
    from kcal*fs*fs/gram/nm to nm: 4.1868e-06
    from kcal*fs/gram/nm to nm/fs: 4.1868e-06
"""

import numpy as np
import os


def init():
    # ----- box properties ------ #

    # initialization
    global ini_x # H2O molecules initially in x, y, z directions
    ini_x = 6
    global ini_y
    ini_y = 6
    global ini_z
    ini_z = 6
    global a_lat # initial latice spacing
    a_lat = sigma

    # bounds
    xlo = 0
    xhi = (ini_x + 1)*a_lat
    ylo = 0
    yhi = (ini_y + 1)*a_lat
    zlo = 0
    zhi = (ini_z + 1)*a_lat
    global bounds
    bounds = [[xlo, xhi], [ylo, yhi], [zlo, zhi]]



    # ------ general variables ------ #

    # number of H2O molecules
    global nmol # total number of molecules
    nmol = ini_x * ini_y * ini_z

    # timestep
    global deltat
    deltat = 0.5 # fs
    
    # temperature
    global Tdesired # desired temperature of the experiment
    Tdesired = 300.0 # K
    global tau # parameter for Berendsen thermostat
    tau = 500*deltat # fs

    # constants and conversion factors
    global kb  # boltzmann's constant
    kb = 0.0019849421 # kcal/mole/K    .......................
    #convdistance = 4.1868e-06
    #convvelocity = 4.1868e-06

    # masses
    massH = 1.6735e-27 # kg
    massO = 2.6561e-26 # kg
    global masses # array might seem inelegant, but speeds up the code
    masses = np.tile([massO, massH, massH], nmol)

    # degrees of freedom per H2O molecule
    global n_dof_mol
    n_dof_mol = 9 # 3*3*3 = 9 translational, bonds and angles not fixed



    # ------ potential parameters ------ #

    # intramolecular potential
    # bonds
    global k_bond
    k_bond = 1.058e5 # kcal/mole/nm^2
    global s0
    s0 = 0.1 # nm
    # angles
    global k_angle
    k_angle = 75.0 # kcal/mole/rad^2
    global theta0
    theta0 = 104.5 * np.pi/180 # rad

    # intermolecular potential
    # Lennard-Jones
    global eps
    eps = 0.2*kb*Tdesired # ...............................................
    global sigma
    sigma = 0.321 # nm
    global cutoff # valid for both, LJ and Coulomb
    cutoff = 2.5*sigma # nm
    # Coulomb
    global qO
    qO = -0.84 # e
    global qH
    qH = 0.42 # e
    global eps0_el
    eps0_el = ... # ........................................................
    global alpha
    alpha = 1/cutoff # 1/nm



    # ------ histogram properties ------ #
    global rmax_hist # maximum distance for histogram
    rmax_hist = 20*sigma # nm
    global dr_hist # width of the histogram bins
    dr_hist = 0.3*sigma # nm



    # ------ data management ------ #

    # frequency to save the data (every n_save timesteps)
    global n_save
    n_save = 10

    # path to the output files
    global path  
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")