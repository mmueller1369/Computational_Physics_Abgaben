import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
import ovito._extensions.pyscript


import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import settings
from ovito.io import import_file
from forces import pbc
from numba import njit

settings.init()
trajc_loc = os.path.join(settings.path, "TrajectoryforC.txt")
trajd_loc = os.path.join(settings.path, "TrajectoryforD.txt")

def read_pos(filename):
    pipeline = import_file(filename)
    n_steps = pipeline.source.num_frames
    data0 = pipeline.compute(0)
    n_particles = data0.particles.count
    x = np.zeros((n_steps, n_particles))
    y = np.zeros((n_steps, n_particles))
    z = np.zeros((n_steps, n_particles))
    for step in range(n_steps):
        data = pipeline.compute(step)
        pos = data.particles['Position'][:]  # shape (n_particles, 3)
        x[step, :] = pos[:, 0]
        y[step, :] = pos[:, 1]
        z[step, :] = pos[:, 2]
    return x, y, z

@njit(parallel=True)
def extract_b(xx, yy, zz):
    b = np.zeros(shape=(len(xx),len(xx[0])//2))
    for t, (x, y, z) in enumerate(zip(xx, yy, zz)):
        bt = np.zeros(shape=(len(x)//2))
        for mol in range(len(x)//2):
            i = mol * 2
            j = mol * 2 + 1
            rijx = pbc(x[i], x[j], settings.xlo, settings.xhi)
            rijy = pbc(y[i], y[j], settings.ylo, settings.yhi)
            rijz = pbc(z[i], z[j], settings.zlo, settings.zhi)
            bm = np.sqrt(rijx * rijx + rijy * rijy + rijz * rijz)
            bt[mol] = bm
        b[t, :] = bt
    return b


# ----------------- Part b ----------------- #
xd, yd, zd = read_pos(trajd_loc)
energies = np.loadtxt(os.path.join(settings.path, 'Energy.txt')).T
r1, gr1 = np.loadtxt(os.path.join(settings.path, 'g(r)_1.txt')).T
r2, gr2 = np.loadtxt(os.path.join(settings.path, 'g(r)_2.txt')).T

# iii Bond length distribution
b = extract_b(xd, yd, zd)
plt.figure()
plt.hist(b.flatten(), bins=200, density=True, label='Distribution', color='b')
plt.axvline(settings.b0, label=r"$b_0$", color='r')
plt.xlabel(r'$b$')
plt.ylabel(r'$P(b)$')
plt.tight_layout()
plt.xlim(settings.b0-.05,settings.b0+.05)
plt.legend()
plt.show()

# iv Time evolution mean(b)
mean_b = np.mean(b, axis=1)
timesteps = energies[0]
plt.figure()
plt.plot(timesteps, mean_b, color='b', label=r'$\langle b\rangle_N$')
plt.axhline(settings.b0, label=r"$b_0$", color='r')
plt.xlabel(r'$t$ [fs]')
plt.ylabel(r'$b$')
plt.tight_layout()
plt.legend()
plt.show()

# v Time evolution Ubond
Ubond_mean = energies[3]/settings.n1**3
plt.figure()
plt.plot(timesteps, Ubond_mean, color='b')
plt.xlabel(r'$t$ [fs]')
plt.ylabel(r'$\langle U_\mathrm{bond}\rangle_N$')
plt.tight_layout()
plt.show()

# vi gr
plt.figure(figsize=(7, 5))
plt.plot(r1, gr1, color="b", label=r"$g(r)$ for...")
plt.plot(r2, gr2, color="r", label=r"$g(r)$ for...")
plt.xlabel(r"$r / \sigma$")
plt.ylabel(r"$g(r)$")
plt.legend()
plt.tight_layout()
# plt.savefig(os.path.join(settings.path, "b_g.png"), bbox_inches="tight", dpi=300)
plt.show()