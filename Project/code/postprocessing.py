import matplotlib.pyplot as plt
import math
import os
import numpy as np
from numba import njit, prange
from ovito.io import import_file
import settings
settings.init()


@njit(parallel=True)
def read_data(filename, mode='position'):
    if mode == 'position':
        add = 0
    if mode == 'velocity':
        add = 3
    if mode == 'force':
        add = 6
    pipeline = import_file(filename)
    n_steps = pipeline.source.num_frames
    data0 = pipeline.compute(0)
    n_particles = data0.particles.count
    steps = np.zeros(n_steps, dtype=np.int64)
    x = np.zeros((n_steps, n_particles))
    y = np.zeros((n_steps, n_particles))
    z = np.zeros((n_steps, n_particles))
    for step in prange(n_steps):
        steps[step] = step
        data = pipeline.compute(step)
        pos = data.particles['Position'][:]  # shape (n_particles, 3)
        x[step, :] = pos[:, 0 + add]
        y[step, :] = pos[:, 1 + add]
        z[step, :] = pos[:, 2 + add]
    return steps, x, y, z


@njit(parallel=True)
def calculate_molecule_properties(x, y, z):
    nmol = x.shape[1] // 3
    steps = x.shape[0]
    sis = np.zeros((steps, nmol))
    sjs = np.zeros((steps, nmol))
    thetas = np.zeros((steps, nmol))
    for step in prange(steps):
        for mol in prange(nmol):
            # identify the indices for the oxygen and hydrogens
            o = 3 * mol
            i = 3 * mol + 1
            j = 3 * mol + 2
            # calculate the properties
            six = x[step,o] - x[step,i]
            siy = y[step,o] - y[step,i]
            siz = z[step,o] - z[step,i]
            sjx = x[step,o] - x[step,j]
            sjy = y[step,o] - y[step,j]
            sjz = z[step,o] - z[step,j]
            si = math.sqrt(six*six + siy*siy + siz*siz)
            sj = math.sqrt(sjx*sjx + sjy*sjy + sjz*sjz)
            sproj = six*sjx + siy*sjy + siz*sjz
            theta = math.acos(sproj/(si*sj))
            # store the properties
            sis[step, mol] = si
            sjs[step, mol] = sj
            thetas[step, mol] = theta
    return sis, sjs, thetas


@njit(parallel=True)
def calculate_com(x_step, y_step, z_step, masses):
    total_mass = np.sum(masses)
    tot_x = 0
    tot_y = 0
    tot_z = 0
    for i in prange(len(masses)):
        tot_x += x_step[i] * masses[i]
        tot_y += y_step[i] * masses[i]
        tot_z += z_step[i] * masses[i]
    com_x = tot_x / total_mass
    com_y = tot_y / total_mass
    com_z = tot_z / total_mass
    return com_x, com_y, com_z


@njit(parallel=True)
def measure_rho(x_step, y_step, z_step, masses, mode = "O"):
    com_x, com_y, com_z = calculate_com(x_step, y_step, z_step, masses)
    if mode == "O":
        # O atoms are at indices 0, 3, 6, ...
        x_at = x_step[::3] - com_x
        y_at = y_step[::3] - com_y
        z_at = z_step[::3] - com_z
    if mode == "H":
        # H atoms are at indices 1,2, 4,5, 7,8, ...
        x_at = np.concatenate((x_step[1::3], x_step[2::3])) - com_x
        y_at = np.concatenate((y_step[1::3], y_step[2::3])) - com_y
        z_at = np.concatenate((z_step[1::3], z_step[2::3])) - com_z

    hist = np.zeros(int(settings.rmax_hist / settings.dr_hist))
    for i in prange(len(x_at)):
        r = math.sqrt(x_at[i]*x_at[i] + y_at[i]*y_at[i] + z_at[i]*z_at[i])
        if r < settings.rmax_hist:
            bin_n = int(r/settings.dr_hist)
            hist[bin_n] += 1

    volumes = np.zeros(int(settings.rmax_hist / settings.dr_hist))
    for i in prange(int(settings.rmax_hist / settings.dr_hist)):
        r = i*settings.dr_hist
        volumes[i] = 4/3*np.pi * ((r + settings.dr_hist)**3 - r**3)
        
    return hist/volumes


@njit(parallel=True)
def calculate_dipole_vectors(x_step, y_step, z_step, masses):
    nmol = len(masses)//3
    px = np.zeros(nmol)
    py = np.zeros(nmol)
    pz = np.zeros(nmol)
    for mol in prange(nmol):
        o = i * 3
        i = o + 1
        j = o + 2

        six = x_step[o] - x_step[i]
        siy = y_step[o] - y_step[i]
        siz = z_step[o] - z_step[i]
        sjx = x_step[o] - x_step[j]
        sjy = y_step[o] - y_step[j]
        sjz = z_step[o] - z_step[j]

        px[mol] = (six + sjx) / 2
        py[mol] = (siy + sjy) / 2
        pz[mol] = (siz + sjz) / 2
    return px, py, pz


@njit(parallel=True)
def calculate_com_molecules(x_step, y_step, z_step, masses):
    comtot_x, comtot_y, comtot_z = calculate_com(x_step, y_step, z_step, masses)
    nmol = len(masses)//3
    coms_x = np.zeros(nmol)
    coms_y = np.zeros(nmol)
    coms_z = np.zeros(nmol)
    for mol in prange(nmol):
        slice_min = mol*3
        slice_max = slice_min + 3
        x, y, z = calculate_com(x_step[slice_min:slice_max],
                                y_step[slice_min:slice_max],
                                z_step[slice_min:slice_max],
                                masses[slice_min:slice_max])
        coms_x[mol] = x - comtot_x
        coms_y[mol] = y - comtot_y
        coms_z[mol] = z - comtot_z
    return coms_x, coms_y, coms_z


@njit(parallel=True)
def calculate_dipole_projections(x_step, y_step, z_step, masses):
    px, py, pz = calculate_dipole_vectors(x_step, y_step, z_step, masses)
    coms_x, coms_y, coms_z = calculate_com_molecules(x_step, y_step, z_step, masses)
    nmol = len(masses)//3
    proj = np.zeros(nmol)
    for mol in prange(nmol):
        r = math.sqrt(coms_x[mol]*coms_x[mol] + coms_y[mol]*coms_y[mol] + coms_z[mol]*coms_z[mol])
        dirx = coms_x / r
        diry = coms_y / r
        dirz = coms_z / r
        projmol = px*dirx + py*diry + pz*dirz
        proj[mol] = projmol
    return proj

        