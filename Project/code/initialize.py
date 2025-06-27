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