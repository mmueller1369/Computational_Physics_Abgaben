import settings
import random
import math
import numpy as np
from tqdm import tqdm


def single(pert_length=0, pert_angle=0):
    x = np.zeros(shape=(3))
    y = np.zeros(shape=(3))
    z = np.zeros(shape=(3))
    vx = np.zeros(shape=(3))
    vy = np.zeros(shape=(3))
    vz = np.zeros(shape=(3))

    length = settings.s0 * + pert_length
    angle = settings.theta0 + pert_angle

    x[1] = settings.s0
    x[2] = length * math.cos(angle)
    y[2] = length * math.sin(angle)

    return x, y, z, vx, vy, vz


def cubic_lattice():
    nmol = settings.n1 * settings.n2 * settings.n3
    natoms = 3 * nmol
    x = np.zeros(natoms)
    y = np.zeros(natoms)
    z = np.zeros(natoms)
    vx = np.zeros(natoms)
    vy = np.zeros(natoms)
    vz = np.zeros(natoms)
    n = 0
    pbar = tqdm(total=nmol, desc="Initializing H2O-molecules")
    for nx in range(settings.n1):
        for ny in range(settings.n2):
            for nz in range(settings.n3):
                # position of oxygen
                ox = nx * settings.a_lat + settings.a_lat / 2.0
                oy = ny * settings.a_lat + settings.a_lat / 2.0
                oz = nz * settings.a_lat + settings.a_lat / 2.0
                # creating first random vector v1
                v1 = random.random(shape=(3))
                v1 /= np.linalg.norm(v1)
                v1 *= settings.s0 * math.cos(settings.theta0/2)
                # creating second random vector which yields v2 orthogonal to v1
                v2_helper = random.random(shape=(3))
                v2 = np.cross(v1, v2_helper)
                v2 /= np.linalg.norm(v2)
                v2 *= settings.s0 * math.sin(settings.theta0/2)
                # set positions of all components in same molecule
                x[n] = ox
                y[n] = oy
                z[n] = oz
                x[n+1] = ox + v1[0] + v2[0]
                y[n+1] = oy + v1[1] + v2[1]
                z[n+1] = oz + v1[2] + v2[2]
                x[n+2] = ox + v1[0] - v2[0]
                y[n+2] = oy + v1[1] - v2[1]
                z[n+2] = oz + v1[2] - v2[2]
                # same initial velocity in same molecule
                vx0 = 0.5 - random.randint(0, 1)
                vy0 = 0.5 - random.randint(0, 1)
                vz0 = 0.5 - random.randint(0, 1)
                vx[n] = vx0
                vy[n] = vy0
                vz[n] = vz0
                vx[n+1] = vx0
                vy[n+1] = vy0
                vz[n+1] = vz0
                vx[n+2] = vx0
                vy[n+2] = vy0
                vz[n+2] = vz0
                n += 3
                pbar.update(1)
    pbar.close()

    # cancel the linear momentum
    vx -= np.sum(vx) / nmol
    vy -= np.sum(vy) / nmol
    vz -= np.sum(vz) / nmol

    # rescale the velocity to the desired temperature
    Trandom = temperature(vx, vy, vz)
    vx, vy, vz = rescalevelocity(vx, vy, vz, Tdesired, Trandom)

    return x, y, z, vx, vy, vz