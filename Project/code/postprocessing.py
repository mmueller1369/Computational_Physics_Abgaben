import matplotlib.pyplot as plt
import math
import os
import numpy as np
from numba import njit, prange
from ovito.io import import_file
import ovito.modifiers as om
import settings
# import settings_SI as settings
settings.init()


def read_data(filename, mode='Position'):
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
        pos = data.particles[mode][:]  # shape (n_particles, 3)
        x[step, :] = pos[:, 0]
        y[step, :] = pos[:, 1]
        z[step, :] = pos[:, 2]
    return steps, x, y, z


def make_pipeline_droplet(filename, cutoff):
    pipeline = import_file(filename)
    pipeline.modifiers.append(om.ClusterAnalysisModifier(cutoff=cutoff,
                                                         sort_by_size=True,
                                                         unwrap_particles=True,
                                                         compute_com=True,
                                                         compute_gyration=True))
    pipeline.modifiers.append(om.ExpressionSelectionModifier(expression="Cluster != 1"))
    pipeline.modifiers.append(om.DeleteSelectedModifier())
    
    def DeleteIncompleteMoleculesModifier(frame, data):
        molIDs = np.unique(data.particles['Molecule Identifier'][...])
        for molID in molIDs:
            if np.sum(data.particles['Molecule Identifier'] == molID) != 3:
                om.ExpressionSelectionModifier(
                    expression=f"Molecule Identifier == {molID}").modify(frame, data)
                om.DeleteSelectedModifier().modify(frame, data)
    pipeline.modifiers.append(DeleteIncompleteMoleculesModifier)

    return pipeline


# ausrichtung by distance
# overall asphericity
# molecule in droplet over time


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
def calculate_com(pipeline, step, data=False):
    if not data:
        data = pipeline.compute(step)
    total_mass = np.sum(data.particles.masses)
    tot_x = 0
    tot_y = 0
    tot_z = 0
    for i in prange(data.particles.count):
        tot_x += data.particles.position.T[0, i] * data.particles.masses[i]
        tot_y += data.particles.position.T[1, i] * data.particles.masses[i]
        tot_z += data.particles.position.T[2, i] * data.particles.masses[i]
    com_x = tot_x / total_mass
    com_y = tot_y / total_mass
    com_z = tot_z / total_mass
    return com_x, com_y, com_z


# @njit(parallel=True)
# def measure_rho_old(pipeline, step, mode = "both"):
#     # com_x, com_y, com_z = calculate_com(pipeline, step)
#     if mode == 'O':
#         pipeline.modifiers.append(om.ExpressionSelectionModifier(expression="ParticleType == 1"))
#         pipeline.modifiers.append(om.DeleteSelectedModifier())
#     elif mode == 'H':
#         pipeline.modifiers.append(om.ExpressionSelectionModifier(expression="ParticleType == 2"))
#         pipeline.modifiers.append(om.DeleteSelectedModifier())
#     data = pipeline.compute(step)
#     com_x, com_y, com_z = data.tables['clusters']['Center of Mass'][0]
#     x_at = data.particles.position.T[0] - com_x
#     y_at = data.particles.position.T[1] - com_y
#     z_at = data.particles.position.T[2] - com_z

#     hist = np.zeros(int(settings.rmax_hist / settings.dr_hist))
#     for i in prange(len(x_at)):
#         r = math.sqrt(x_at[i]*x_at[i] + y_at[i]*y_at[i] + z_at[i]*z_at[i])
#         if r < settings.rmax_hist:
#             bin_n = int(r/settings.dr_hist)
#             hist[bin_n] += 1

#     volumes = np.zeros(int(settings.rmax_hist / settings.dr_hist))
#     for i in prange(int(settings.rmax_hist / settings.dr_hist)):
#         r = i*settings.dr_hist
#         volumes[i] = 4/3*np.pi * ((r + settings.dr_hist)**3 - r**3)

#     return hist/volumes


@njit(parallel=True)
def calculate_dipole_vectors(pipeline, step):
    data = pipeline.compute(step)
    nmol = data.particles.count//3
    px = np.zeros(nmol)
    py = np.zeros(nmol)
    pz = np.zeros(nmol)
    for mol in prange(nmol):
        o = i * 3
        i = o + 1
        j = o + 2

        six = data.particles.position.T[0,o] - data.particles.position.T[0,i]
        siy = data.particles.position.T[1,o] - data.particles.position.T[1,i]
        siz = data.particles.position.T[2,o] - data.particles.position.T[2,i]
        sjx = data.particles.position.T[0,o] - data.particles.position.T[0,j]
        sjy = data.particles.position.T[1,o] - data.particles.position.T[1,j]
        sjz = data.particles.position.T[2,o] - data.particles.position.T[2,j]

        px[mol] = (six + sjx) / 2
        py[mol] = (siy + sjy) / 2
        pz[mol] = (siz + sjz) / 2
    return px, py, pz


@njit(parallel=True)
def calculate_com_molecules(pipeline, step):
    data = pipeline.compute(step)
    comtot_x, comtot_y, comtot_z = data.tables['clusters']['Center of Mass'][0]
    nmol = data.particles.count // 3
    coms_x = np.zeros(nmol)
    coms_y = np.zeros(nmol)
    coms_z = np.zeros(nmol)
    for mol in prange(nmol):
        slice_min = mol*3
        slice_max = slice_min + 3
        x, y, z = calculate_com(data.particles.position.T[0,slice_min:slice_max],
                                data.particles.position.T[1,slice_min:slice_max],
                                data.particles.position.T[2,slice_min:slice_max],
                                data.particles.masses[slice_min:slice_max])
        coms_x[mol] = x - comtot_x
        coms_y[mol] = y - comtot_y
        coms_z[mol] = z - comtot_z
    return coms_x, coms_y, coms_z


@njit(parallel=True)
def calculate_distances(pipeline, step):
    coms_x, coms_y, coms_z = calculate_com_molecules(pipeline, step)

    dist = np.zeros(len(coms_x))
    for i in prange(len(dist)):
        r = math.sqrt(coms_x[i]*coms_x[i] + coms_y[i]*coms_y[i] + coms_z[i]*coms_z[i])
        dist[i] = r
        
    return dist


@njit(parallel=True)
def measure_rho_step(pipeline, step):
    dist = calculate_distances(pipeline, step)
    hist = np.zeros(int(settings.rmax_hist / settings.dr_hist))
    for i in prange(len(dist)):
        r = dist[i]
        if r < settings.rmax_hist:
            bin_n = int(r/settings.dr_hist)
            hist[bin_n] += 1

    volumes = np.zeros(int(settings.rmax_hist / settings.dr_hist))
    for i in prange(int(settings.rmax_hist / settings.dr_hist)):
        r = i*settings.dr_hist
        volumes[i] = 4/3*np.pi * ((r + settings.dr_hist)**3 - r**3)

    return hist/volumes


@njit(parallel=True)
def measure_rho_all(pipeline, step):
    steps = pipeline.frames.last_frame
    hist = np.zeros(int(settings.rmax_hist / settings.dr_hist))
    for step in steps:
        dist = calculate_distances(pipeline, step)
        for i in prange(len(dist)):
            r = dist[i]
            if r < settings.rmax_hist:
                bin_n = int(r/settings.dr_hist)
                hist[bin_n] += 1

    volumes = np.zeros(int(settings.rmax_hist / settings.dr_hist))
    for i in prange(int(settings.rmax_hist / settings.dr_hist)):
        r = i*settings.dr_hist
        volumes[i] = 4/3*np.pi * ((r + settings.dr_hist)**3 - r**3)

    return hist/volumes


@njit(parallel=True)
def calculate_dipole_projections(pipeline, step):
    px, py, pz = calculate_dipole_vectors(pipeline, step)
    coms_x, coms_y, coms_z = calculate_com_molecules(pipeline, step)
    nmol = len(coms_x)//3
    proj = np.zeros(nmol)
    for mol in prange(nmol):
        r = math.sqrt(coms_x[mol]*coms_x[mol] + coms_y[mol]*coms_y[mol] + coms_z[mol]*coms_z[mol])
        dirx = coms_x / r
        diry = coms_y / r
        dirz = coms_z / r
        projmol = px*dirx + py*diry + pz*dirz
        proj[mol] = projmol
    return proj


@njit(parallel=True)
def calculate_dipole_projections_by_distance_step(pipeline, step):
    proj = calculate_dipole_projections(pipeline, step)
    dist = calculate_distances(pipeline, step)
    hist = np.zeros(int(settings.rmax_hist / settings.dr_hist))
    hist_count = np.zeros(int(settings.rmax_hist / settings.dr_hist))
    for i in prange(len(dist)):
        r = dist[i]
        if r < settings.rmax_hist:
            bin_n = int(r/settings.dr_hist)
            hist[bin_n] += proj[i]
            hist_count[bin_n] += 1

    return hist/hist_count


@njit(parallel=True)
def calculate_dipole_projections_by_distance_all(pipeline, step=False):
    hist = np.zeros(int(settings.rmax_hist / settings.dr_hist))
    hist_count = np.zeros(int(settings.rmax_hist / settings.dr_hist))
    steps = pipeline.frames.last_frame
    for step in steps:
        proj = calculate_dipole_projections(pipeline, step)
        dist = calculate_distances(pipeline, step)
        for i in prange(len(dist)):
            r = dist[i]
            if r < settings.rmax_hist:
                bin_n = int(r/settings.dr_hist)
                hist[bin_n] += proj[i]
                hist_count[bin_n] += 1

    return hist/hist_count