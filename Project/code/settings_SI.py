"""
--------- units ---------
length: nm
time: fs
mass: gram/mole
velocity: nm/fs
force: (kcal/mole)/nm
energy: kcal/mole
temperature: K
angle: rad
charge: elementary charge e
"""

import numpy as np
import os


def init():
    # ----- simulation properties ------ #

    # timestep
    global deltat
    deltat = .5e-15 # 0.5 # fs
    
    # temperature
    global Tdesired # desired temperature of the experiment
    Tdesired = 300.0 # K
    global tau # parameter for Berendsen thermostat
    tau = 500*deltat # fs
    global kb  # boltzmann's constant
    kb = 1.38e-23 # 1.9872036e-3 # kcal/mole/K
    N_A = 6.022140857e23
    e = 1.602176621e-19

    # initialization
    global ini_x # H2O molecules initially in x, y, z directions
    ini_x = 6
    global ini_y
    ini_y = ini_x
    global ini_z
    ini_z = ini_x
    global a_lat # initial latice spacing
    a_lat = 3.21e-10 # 0.321 # = sigma, nm

    # molecule properties
    global nmol # total number of molecules
    nmol = ini_x * ini_y * ini_z
    global rho
    rho = 1  # molecules/sigma^-3
    global masses # array might seem inelegant, but speeds up the code
    massH = 1.6735e-27 # 1.007805272 # g/mole
    massO = 2.6561e-26 # 15.99540833 # g/mole
    masses = np.tile([massO, massH, massH], nmol)

    # bounds
    xlo = 0
    xhi = ini_x * a_lat
    ylo = 0
    yhi = ini_y * a_lat
    zlo = 0
    zhi = ini_z * a_lat
    global bounds
    bounds = np.array([[xlo, xhi], [ylo, yhi], [zlo, zhi]])



    # ------ potential parameters ------ #

    # intramolecular potential
    ## bonds
    global k_bond
    k_bond = 4184 / N_A / 1e-20 * 1058.0 # 1.058e5 # kcal/mole/nm^2
    global s0
    s0 = 0.1e-9 # 0.1 # nm
    ## angles
    global k_angle
    k_angle = 4184 / N_A * 75.0 # 75.0 # kcal/mole/rad^2
    global theta0
    theta0 = 104.5 * np.pi/180 # rad

    # intermolecular potential
    ## Lennard-Jones
    global eps
    eps = 0.2*kb*Tdesired # kcal/mole
    global sigma
    sigma = 3.21e-10 # 0.321 # nm
    global cutoff # valid for both, LJ and Coulomb
    cutoff = 2.5*sigma # nm
    ## Coulomb
    global qO
    qO = -0.84 * e # -0.84 # e
    global qH
    qH = 0.42 * e # 0.42 # e
    global eps0_el
    eps0_el = 8.8541878188e-12 # 2.39451898e-3 # e^2/(kcal/mole)/nm
    global alpha
    alpha = 1/cutoff # 1/nm



    # ------  conversion factor ------ #
    global conv_factor # from g/mole*nm^2/fs^2 to kcal/mole (energies)
                       # or from g/mole*nm/fs^2 to kcal/mole/nm (forces)
    conv_factor = 1 # 2.390057361e5
    global conv_length
    conv_length = 1e9 # m to nm
    global conv_velocity
    conv_velocity = 1e-6 # 1e9 / 1e15 # m/s to nm/fs
    global conv_mass
    conv_mass = 1e3 * N_A # kg to g/mole
    global conv_energy
    conv_energy = 1 / 4184 * N_A # J to kcal/mole
    global conv_force
    conv_force = conv_energy / conv_length # J/m to kcal/mole/nm



    # ------ histogram properties ------ #
    global rmax_hist # maximum distance for histogram
    rmax_hist = xhi / 2 # nm
    global dr_hist # width of the histogram bins
    dr_hist = 0.3*sigma # nm



    # ------ data management ------ #

    # frequency to save the data (every n_save timesteps)
    global n_save
    n_save = 10

    # path to the output files
    global path  
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")