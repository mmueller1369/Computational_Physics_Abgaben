import numpy as np
import math
import matplotlib.pyplot as plt
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
    masses = settings.masses[:3]
    x_com = np.sum(x*masses) / np.sum(masses)
    y_com = np.sum(y*masses) / np.sum(masses)
    z_com = np.sum(z*masses) / np.sum(masses)
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
    mol2_flipped = mol2.copy()
    mol2_flipped[:, :3] *= -1  # flip x and y-coordinate of mol2
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            if x**2 + y**2 + offset**2 >= minimal_distance**2:  # only calculate within circle
                if y >= 0:
                    mol = mol2
                else:
                    mol = mol2_flipped
                mol2_shifted = mol + np.array([x, y, 0.0])
                grid[j, i] = calculate_energy(mol1, mol2_shifted)
            else:
                grid[j, i] = np.nanargmax(grid)
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

def heatmap(values, angle, offset, maximal_distance, cbar_boundaries, ax=None, mark_second_molecule=True, legend=True):
    min_value, max_value = cbar_boundaries
    v1 = np.array([1.0,0.0,0.0])*settings.s0*math.cos(settings.theta0/2)
    v21 = np.array([0.0, 1.0, 0.0])*settings.s0*math.sin(settings.theta0/2)
    mol1 = create_molecule(v1, v21)
    masses = settings.masses[0:3]
    total_mass = np.sum(masses)
    com_x = np.sum(mol1[:,0] * masses) / total_mass
    com_y = np.sum(mol1[:,1] * masses) / total_mass
    O_x, O_y = mol1[0,0]/settings.sigma, mol1[0,1]/settings.sigma
    H1_x, H1_y = mol1[1,0]/settings.sigma, mol1[1,1]/settings.sigma
    H2_x, H2_y = mol1[2,0]/settings.sigma, mol1[2,1]/settings.sigma
    ddx = (com_x-H1_x)/2

    # formatting stuff
    marker_size = 6  # Standardgröße ist 6, also doppelt
    lw_flat = 1
    lw_upper = 2
    lw_lower = 0.5
    color_o = 'red'
    color_h = 'white'
    color_com = 'black'
    color_bond = 'gray'
    show_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        show_fig = True

    # Marker für O, H1, H2, COM
    # Erst Linien, dann Marker
    ax.plot([H1_x, O_x, H2_x], [H1_y, O_y, H2_y], lw=lw_flat, color=color_bond, zorder=1)
    ax.scatter([O_x], [O_y], c=color_o, marker='o', label='O', s=marker_size**2*4, edgecolors='black', linewidths=.5, zorder=2)
    ax.scatter([H1_x, H2_x], [H1_y, H2_y], c=color_h, marker='o', edgecolors='black', linewidths=.5, label='H', s=marker_size**2, zorder=2)
    ax.scatter([com_x], [com_y], c=color_com, marker='*', label='COM', s=marker_size**2/4, zorder=2)
    
    if mark_second_molecule:
        rad = angle*np.pi/180
        v22 = np.array([0.0, math.cos(rad), math.sin(rad)])*settings.s0*math.sin(settings.theta0/2)
        mol2 = create_molecule(v1, v22)
        o_x, o_y = mol2[0,0]/settings.sigma, mol2[0,1]/settings.sigma
        h1_x, h1_y, h1_z = mol2[1,0]/settings.sigma, mol2[1,1]/settings.sigma, mol2[1,2]/settings.sigma
        h2_x, h2_y = mol2[2,0]/settings.sigma, mol2[2,1]/settings.sigma
        for dx, dy, flip in [[1, 1, 1], [1, -1, -1]]:
            if h1_z > 0:
                lw1 = lw_upper
                lw2 = lw_lower
                ls1 = '-'
                ls2 = '--'
                zo1 = 3
                zo2 = 1
            if h1_z < 0:
                lw1 = lw_lower
                lw2 = lw_upper
                ls1 = '--'
                ls2 = '-'
                zo1 = 1
                zo2 = 3
            if h1_z == 0:
                lw1 = lw_flat
                lw2 = lw_flat
                ls1 = '-'
                ls2 = '-'
                zo1 = 3
                zo2 = 3
            # Erst Linien, dann Marker
            ax.plot([(h1_x+ddx)*flip+dx, (o_x+ddx)*flip+dx], [h1_y*flip+dy, o_y*flip+dy], lw=lw1, ls=ls1, color=color_bond, zorder=(zo1+2)/2)
            ax.plot([(h2_x+ddx)*flip+dx, (o_x+ddx)*flip+dx], [h2_y*flip+dy, o_y*flip+dy], lw=lw2, ls=ls2, color=color_bond, zorder=(zo2+2)/2)
            ax.scatter([(o_x+ddx)*flip+dx], [o_y*flip+dy], c=color_o, marker='o', s=marker_size**2*4, edgecolors='black', linewidths=.5, zorder=(zo1+zo2)/2)
            ax.scatter([(h1_x+ddx)*flip+dx], [h1_y*flip+dy], c=color_h, edgecolors='black', linewidths=.5, s=marker_size**2, zorder=zo1)
            ax.scatter([(h2_x+ddx)*flip+dx], [h2_y*flip+dy], c=color_h, edgecolors='black', linewidths=.5, s=marker_size**2, zorder=zo2)
            ax.scatter([(com_x+ddx)*flip+dx], [com_y*flip+dy], c=color_com, marker='*', s=marker_size**2/4, zorder=3)

    ext = maximal_distance/settings.sigma
    im = ax.imshow(values, extent=(-ext, ext, -ext, ext),
                origin='lower', cmap='viridis',
                vmin=min_value, vmax=max_value)
    ax.plot([-ext, ext], [0, 0], color='black', lw=1)
    if legend:
        ax.legend(loc='upper left')
    if show_fig:
        plt.colorbar(im, ax=ax, label=r'$E$ (kcal/mol)')
        ax.set_xlabel(r'$x/\sigma$')
        ax.set_ylabel(r'$y/\sigma$')
        ax.set_title(rf'Angle: {angle:.1f}°, Offset: {offset/settings.sigma:.2f} $\sigma$')
        plt.show()

angles = [0, 45, -45, 90]

def plot(angles, n_offsets=3, n_grid=10, maximal_distance=2.5*settings.sigma, minimal_distance=0.5*settings.sigma):
    nangles = len(angles)
    offsets = np.linspace(0, maximal_distance, n_offsets)
    values = calculate_values(angles, offsets, n_grid, maximal_distance, minimal_distance)
    max_value = min(np.nanmax(values), 1)
    min_value = np.nanmin(values)
    cbar_boundaries = [min_value, max_value]

    fig, axes = plt.subplots(nangles+1, n_offsets+1, figsize=(2.5*(n_offsets+1), 2.5*(nangles+1)), squeeze=True)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.tight_layout(pad=0)

    # Indizierung: oberste Zeile = Offset, linke Spalte = Angle
    for i in range(1, nangles+1):
        axes[i,0].axis('off')
        axes[i,0].text(0.5, 0.5, f'angle\n{angles[i-1]:.1f}°',
                      ha='center', va='center', fontsize=10, transform=axes[i,0].transAxes)
    for j in range(1, n_offsets+1):
        axes[0,j].axis('off')
        axes[0,j].text(0.5, 0.5, f'offset\n{offsets[j-1]/settings.sigma:.2f} $\\sigma$',
                      ha='center', va='center', fontsize=10, transform=axes[0,j].transAxes)
    # Linkes oberes Feld: Richtungen und Colorbar
    ax00 = axes[0,0]
    ax00.axis('off')
    # Dummy-Heatmap für Colorbar
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax00, orientation='horizontal', fraction=0.4, pad=0, aspect=10)
    cbar.set_label(r'$E$ (kcal/mol)')

    # Heatmaps
    for i in range(1, nangles+1):
        for j in range(1, n_offsets+1):
            mark_second_molecule, legend = False, False
            if j == 3:
                mark_second_molecule = True
                if i == 1:
                    legend = True
            heatmap(values[i-1, j-1], angles[i-1], offsets[j-1], maximal_distance, cbar_boundaries, ax=axes[i,j], mark_second_molecule=mark_second_molecule, legend=legend)
            axes[i,j].set_title("")
            axes[i,j].set_xlabel("")
            axes[i,j].set_ylabel("")
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])

    plt.tight_layout()
    plt.show()

plot(angles, n_grid=100, maximal_distance=1.5*settings.sigma)