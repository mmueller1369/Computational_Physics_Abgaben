import matplotlib.pyplot as plt
import math
import os
import numpy as np
from numba import njit, prange
from ovito.io import import_file
import settings
settings.init()

loc = os.path.join(settings.path, "TrajectoryforC.txt")

def read_pos(filename):
    pipeline = import_file(filename)
    n_steps = pipeline.source.num_frames
    data0 = pipeline.compute(0)
    n_particles = data0.particles.count
    steps = np.zeros(n_steps, dtype=np.int64)
    x = np.zeros((n_steps, n_particles))
    y = np.zeros((n_steps, n_particles))
    z = np.zeros((n_steps, n_particles))
    for step in range(n_steps):
        steps[step] = step
        data = pipeline.compute(step)
        pos = data.particles['Position'][:]  # shape (n_particles, 3)
        x[step, :] = pos[:, 0]
        y[step, :] = pos[:, 1]
        z[step, :] = pos[:, 2]
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