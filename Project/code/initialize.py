import random
import math
import numpy as np
from tqdm import tqdm
import tools
import settings
# import settings_SI as settings
settings.init()


def single(pert_length=0, pert_angle=0):
    x = np.zeros(shape=(3))
    y = np.zeros(shape=(3))
    z = np.zeros(shape=(3))
    vx = np.zeros(shape=(3))
    vy = np.zeros(shape=(3))
    vz = np.zeros(shape=(3))

    length = settings.s0 + pert_length
    angle = settings.theta0 + pert_angle

    x[1] = settings.s0
    x[2] = length * math.cos(angle)
    y[2] = length * math.sin(angle)

    return x, y, z, vx, vy, vz


def cubic_lattice():
    natoms = 3 * settings.nmol
    x = np.zeros(natoms)
    y = np.zeros(natoms)
    z = np.zeros(natoms)
    vx = np.zeros(natoms)
    vy = np.zeros(natoms)
    vz = np.zeros(natoms)
    n = 0
    pbar = tqdm(total=settings.nmol, desc="Initializing H2O molecules")
    for nx in range(settings.ini_x):
        for ny in range(settings.ini_y):
            for nz in range(settings.ini_z):
                # position of oxygen
                ox = nx * settings.a_lat + settings.a_lat / 2.0
                oy = ny * settings.a_lat + settings.a_lat / 2.0
                oz = nz * settings.a_lat + settings.a_lat / 2.0
                # creating first random vector v1
                v1 = np.random.rand(3)
                v1 *= random.choice([-1, 1]) # randomize sign for full sphere
                # v1 = np.array([1.0,0.0,0.0])
                v1 /= np.linalg.norm(v1)
                v1 *= settings.s0 * math.cos(settings.theta0/2)
                # creating second random vector which yields v2 orthogonal to v1
                v2_helper = np.random.rand(3)
                v2_helper *= random.choice([-1, 1])
                # v2_helper = np.array([0.0,1.0,0.0])
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
                vx0 = 0.5 - np.random.rand()
                vy0 = 0.5 - np.random.rand()
                vz0 = 0.5 - np.random.rand()
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
    total_mass = np.sum(settings.masses)
    vx_com = np.sum(vx * settings.masses) / total_mass
    vy_com = np.sum(vy * settings.masses) / total_mass
    vz_com = np.sum(vz * settings.masses) / total_mass
    vx -= vx_com
    vy -= vy_com
    vz -= vz_com

    # rescale the velocity to the desired temperature
    Trandom = tools.computeTemperature(vx, vy, vz, settings.masses)
    vx, vy, vz = tools.rescaleVelocity(vx, vy, vz, settings.Tdesired, Trandom)

    return x, y, z, vx, vy, vz