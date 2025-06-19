import matplotlib.pyplot as plt
import os
import numpy as np
import settings
from ovito.io import import_file
from forces import pbc
from numba import prange
from scipy.optimize import curve_fit
from tqdm import tqdm

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

# ------------------------------------------ #
# ----------------- Part b ----------------- #
# ------------------------------------------ #
xd, yd, zd = read_pos(trajd_loc)
energies = np.loadtxt(os.path.join(settings.path, 'Energy.txt')).T
r1, gr1 = np.loadtxt(os.path.join(settings.path, 'g(r)_1.txt')).T
r2, gr2 = np.loadtxt(os.path.join(settings.path, 'g(r)_2.txt')).T

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

# iii Bond length distribution
b = extract_b(xd, yd, zd)
plt.figure(figsize=(7, 5))
plt.hist(b.flatten(), bins=200, density=True, label='Distribution', color='b')
plt.axvline(settings.b0, label=r"$b_0$", color='r')
plt.xlabel(r'$b$')
plt.ylabel(r'$P(b)$')
plt.tight_layout()
plt.xlim(settings.b0-.05,settings.b0+.05)
plt.legend()
plt.savefig(os.path.join(settings.path, "b_iii.png"), bbox_inches="tight", dpi=300)

# iv Time evolution mean(b)
mean_b = np.mean(b, axis=1)
timesteps = energies[0]
plt.figure(figsize=(7, 5))
plt.plot(timesteps, mean_b, color='b', label=r'$\langle b\rangle_N$')
plt.axhline(settings.b0, label=r"$b_0$", color='r')
plt.xlabel(r'$t$ [fs]')
plt.ylabel(r'$b$')
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(settings.path, "b_iv.png"), bbox_inches="tight", dpi=300)

# v Time evolution Ubond
Ubond_mean = energies[3]/settings.n1**3
plt.figure(figsize=(7, 5))
plt.plot(timesteps, Ubond_mean, color='b', label=r'$\langle U_\mathrm{bond}\rangle_N$')
plt.axhline(0.5*settings.kb*settings.Tdesired, label=r'$1/2\,k_BT$', color='r')
plt.xlabel(r'$t$ [fs]')
plt.ylabel(r'$U$')
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(settings.path, "b_v.png"), bbox_inches="tight", dpi=300)

# vi g(r)
plt.figure(figsize=(7, 5))
plt.plot(r1*settings.sigma, gr1*4, color="b", label=r"$g(r)$ for the COMs")
plt.plot(r2*settings.sigma, gr2, color="r", label=r"$g(r)$ for the atoms")
plt.ylim(0,5)
plt.xlabel(r"$r$ [nm]")
plt.ylabel(r"$g(r)$")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(settings.path, "b_vi.png"), bbox_inches="tight", dpi=300)


# ------------------------------------------ #
# ----------------- Part c ----------------- #
# ------------------------------------------ #
xc, yc, zc = read_pos(trajd_loc)
energies = np.loadtxt(os.path.join(settings.path, 'Energy.txt')).T
timesteps = energies[0]

def extract_omega(xx, yy, zz):
    n_steps = len(xx)
    n_atoms = len(xx[0])
    n_mol = n_atoms // 2
    omega = np.zeros((n_steps, n_mol, 3))
    for t in prange(n_steps):
        x, y, z = xx[t], yy[t], zz[t]
        for mol in prange(n_mol):
            i = mol * 2
            j = mol * 2 + 1
            rijx = pbc(x[i], x[j], settings.xlo, settings.xhi)
            rijy = pbc(y[i], y[j], settings.ylo, settings.yhi)
            rijz = pbc(z[i], z[j], settings.zlo, settings.zhi)
            norm = np.sqrt(rijx**2 + rijy**2 + rijz**2)
            omega[t, mol, :] = [rijx/norm, rijy/norm, rijz/norm]
    return omega

def extract_ct(omega):
    n_steps = omega.shape[0]
    ct = np.zeros(n_steps)
    for dt in tqdm(range(n_steps)):
        products = np.sum(omega[dt:] * omega[:n_steps-dt], axis=2)  # shape: (n_steps-dt, n_mol)
        ct[dt] = np.mean(products)
    return ct

# i+ii ACF
omega = extract_omega(xc, yc, zc)
ct = extract_ct(omega)
np.savetxt(os.path.join(settings.path, "ct.txt"), ct)
ct = np.loadtxt(os.path.join(settings.path, "ct.txt"))
def exp_decay(t, tau):
    return np.exp(-t / tau)
fit_mask = ct > 0.1 # don't fit too small values
popt, pcov = curve_fit(exp_decay, timesteps[fit_mask], ct[fit_mask], p0=[1000])
tau_R = popt[0]

plt.figure(figsize=(7,5))
plt.plot(timesteps, ct, label=r'$C(t)$', color='b')
plt.plot(timesteps, exp_decay(timesteps, tau_R), 'r--', label=fr'Fit: $\tau_R$ = {tau_R:.1f} fs')
plt.xlabel(r'$t$ [fs]')
plt.ylabel(r'$C(t)$')
plt.legend()
plt.xlim(0,10000)
plt.tight_layout()
plt.savefig(os.path.join(settings.path, "c.png"), bbox_inches="tight", dpi=300)

omega_harm = np.sqrt(settings.kb_di/settings.mass)
print(f"tau_R = {tau_R}")
print(f"tau_harm = {1/omega_harm}")


# ------------------------------------------ #
# ----------------- Part b ----------------- #
# ------------------------------------------ #
xd, yd, zd = read_pos(trajd_loc)
energies = np.loadtxt(os.path.join(settings.path, 'Energy.txt')).T
timesteps = energies[0]

def calculate_com(x, y, z):
    n_mol = len(x) // 2
    com_x = np.zeros(n_mol)
    com_y = np.zeros(n_mol)
    com_z = np.zeros(n_mol)
    for i in prange(n_mol):
        com_x[i] = 0.5 * (x[2 * i] + x[2 * i + 1])
        com_y[i] = 0.5 * (y[2 * i] + y[2 * i + 1])
        com_z[i] = 0.5 * (z[2 * i] + z[2 * i + 1])
    return com_x, com_y, com_z

def unwrap_positions(positions, box_length):
    """
    Unwraps particle trajectories to account for periodic boundary crossings.
    Each time a jump is detected, all subsequent positions are shifted by +/- box_length.
    """
    unwrapped = positions.copy()
    # Calculate jumps for all steps and particles
    delta = unwrapped[1:] - unwrapped[:-1]
    jumps = np.zeros_like(delta)
    jumps[delta < -0.5 * box_length] = 1
    jumps[delta > 0.5 * box_length] = -1
    # Cumulative sum of jumps for each particle
    shift = np.cumsum(jumps, axis=0)
    shift = np.vstack([np.zeros((1, shift.shape[1])), shift])  # prepend zeros for t=0
    unwrapped += shift * box_length
    return unwrapped

def extract_msd(xx, yy, zz):
    n_steps = len(xx)
    xx = unwrap_positions(xx, settings.xhi - settings.xlo)
    yy = unwrap_positions(yy, settings.yhi - settings.ylo)
    zz = unwrap_positions(zz, settings.zhi - settings.zlo)
    com0 = calculate_com(xx[0], yy[0], zz[0])
    msd = np.zeros(n_steps)
    for t in prange(n_steps):
        com_t = calculate_com(xx[t], yy[t], zz[t])
        # no pbcs since we are calculating the MSD
        dx = com_t[0] - com0[0]
        dy = com_t[1] - com0[1]
        dz = com_t[2] - com0[2]
        msd[t] = np.mean(dx**2 + dy**2 + dz**2)
    return msd

# visualization of unwrapper
plt.figure(figsize=(7, 5))
plt.plot(xd[:,0], color='b', label='Original positions')
plt.plot(unwrap_positions(xd, settings.xhi - settings.xlo)[:,0], color='r', label='Unwrapped positions')
plt.axhline(y=settings.xlo, color='black', label='Edges')
plt.axhline(y=settings.xhi, color='black')
plt.xlabel(r'$t$ [fs]')
plt.ylabel(r'$x$ [nm]')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(settings.path, "d_unwrap.png"), bbox_inches="tight", dpi=300)

# MSD calculation
msd = extract_msd(xd, yd, zd)
plt.figure(figsize=(7, 5))
plt.plot(timesteps, msd, color='b')
plt.loglog()
plt.xlabel(r'$t$ [fs]')
plt.ylabel(r'$MSD$ [nm$^2$]')
plt.tight_layout()
plt.grid()
plt.savefig(os.path.join(settings.path, "d_msd.png"), bbox_inches="tight", dpi=300)

# Diffusion calculation
D = msd/6/timesteps
D_final = D[-1]
plt.figure(figsize=(7, 5))
plt.plot(timesteps, D, color='b', label=r'$D(t)$')
plt.axhline(y=D_final, color='r', label=f'$D_{{final}} = {D_final*1e5:.2f} \cdot 10^{{-5}}\,$nm$^2$/fs')
plt.xlabel(r'$t$ [fs]')
plt.ylabel(r'$D(t)$ [nm$^2$/fs]')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(settings.path, "d_d.png"), bbox_inches="tight", dpi=300)
