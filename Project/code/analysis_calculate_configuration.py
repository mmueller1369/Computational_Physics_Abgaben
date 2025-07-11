import numpy as np
import math
import matplotlib.pyplot as plt
import postprocessing
import forces
import settings
# import settings_SI as settings
settings.init()

def create_molecule(v1, v2):
    # creates a molecule with COM at zero
    pos = np.zeros((3,3))
    pos[1] = v1 + v2
    pos[2] = v1 - v2
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    x_com, y_com, z_com = postprocessing.calculate_com(x, y, z, settings.masses[0:3])
    pos[:,0] -= x_com
    pos[:,1] -= y_com
    pos[:,2] -= z_com
    return pos

def calculate_energy(mol1, mol2):
    # calculates the energy between two molecules
    gamma_cut = forces.gamma(settings.cutoff, settings.alpha)
    e_tot = 0.0
    charges = [settings.qO, settings.qH, settings.qH]
    for i in range(3):
        q1 = charges[i]
        for j in range(3):
            q2 = charges[j]
            rol = np.linalg.norm(mol1[i] - mol2[j])
            if rol < settings.cutoff:
                if i == 0 and j == 0:  # O-O interaction
                    _, e_LJ = forces.ffe_LJ(rol, settings.sigma, settings.eps)
                else:  # H-O or H-H interaction
                    e_LJ = 0
                _, e_coul = forces.ffe_coul(rol, q1, q2, settings.eps0_el,
                                            settings.alpha, settings.cutoff, gamma_cut)
                e_tot += e_LJ + e_coul
    return e_tot

def calculate_grid(mol1, mol2, n_grid, maximal_distance, minimal_distance, offset):
    grid = np.full((n_grid, n_grid), np.nan)
    # grid = np.zeros((n_grid, n_grid))
    grid_x = np.linspace(-maximal_distance, maximal_distance, n_grid)
    grid_y = np.linspace(-maximal_distance, maximal_distance, n_grid)
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            if x**2 + y**2 + offset*2 >= minimal_distance**2:  # only calculate within circle
                mol2_shifted = mol2 + np.array([x, y, 0.0])
                grid[i, j] = calculate_energy(mol1, mol2_shifted)
    return grid

def calculate_values(angles, offsets, n_grid, maximal_distance, minimal_distance):
    v1 = np.array([1.0,0.0,0.0])*settings.s0*math.cos(settings.theta0/2)
    v21 = np.array([0.0, 1.0, 0.0])*settings.s0*math.sin(settings.theta0/2)
    mol1 = create_molecule(v1, v21)
    values = np.zeros((len(angles), len(offsets), n_grid, n_grid))
    for a, angle in enumerate(angles):
        rad = angle*np.pi/180
        v22 = np.array([0.0, math.cos(rad), math.sin(rad)])*settings.s0*math.sin(settings.theta0/2)
        mol2_pure = create_molecule(v1, v22)
        for o, off in enumerate(offsets):
            mol2 = mol2_pure + np.array([0.0, 0.0, 1.0])*off
            values[a, o] = calculate_grid(mol1, mol2, n_grid, maximal_distance, minimal_distance, off)
    return values

def plot_values(n_angles=3, n_offsets=5, n_grid=10, maximal_distance=2.5*settings.sigma, minimal_distance=0.5*settings.sigma):
    angles = np.linspace(0, 90, n_angles)
    offsets = np.linspace(0, maximal_distance, n_offsets)
    values = calculate_values(angles, offsets, n_grid, maximal_distance, minimal_distance)
    max_value = min(np.nanmax(values), 1)
    min_value = np.nanmin(values)
    # Molekül plotten
    v1 = np.array([1.0,0.0,0.0])*settings.s0*math.cos(settings.theta0/2)
    v21 = np.array([0.0, 1.0, 0.0])*settings.s0*math.sin(settings.theta0/2)
    mol1 = create_molecule(v1, v21)
    masses = settings.masses[0:3]
    total_mass = np.sum(masses)
    com_x = np.sum(mol1[:,0] * masses) / total_mass
    com_y = np.sum(mol1[:,1] * masses) / total_mass
    O_x, O_y = mol1[0,0], mol1[0,1]
    H1_x, H1_y = mol1[1,0], mol1[1,1]
    H2_x, H2_y = mol1[2,0], mol1[2,1]
    for a, angle in enumerate(angles):
        for o, offset in enumerate(offsets):
            ext = maximal_distance/settings.sigma
            # Marker für O, H1, H2, COM
            plt.scatter([O_x/settings.sigma], [O_y/settings.sigma], c='blue', marker='o', label='O')
            plt.scatter([H1_x/settings.sigma], [H1_y/settings.sigma], c='red', marker='^', label='H')
            plt.scatter([H2_x/settings.sigma], [H2_y/settings.sigma], c='red', marker='^')
            plt.scatter([com_x/settings.sigma], [com_y/settings.sigma], c='black', marker='*', label='COM')
            plt.imshow(values[a, o], extent=(-ext, ext,
                                             -ext, ext),
                       origin='lower', cmap='viridis',
                       vmin=min_value, vmax=max_value)
            plt.colorbar(label=r'$E$ (kcal/mol)')
            plt.title(rf'Angle: {angle:.1f}°, Offset: {offset/settings.sigma:.2f} $\sigma$')
            plt.xlabel(r'$x/\sigma$')
            plt.ylabel(r'$y/\sigma$')
            plt.legend(loc='upper right')
            plt.show()

plot_values(n_angles=5, n_offsets=5, n_grid=100, maximal_distance=1.5*settings.sigma)