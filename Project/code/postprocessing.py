import math
from tqdm import tqdm
import numpy as np
from ovito.io import import_file
import ovito.modifiers as om
import ovito.data as od 
import settings
# import settings_SI as settings
settings.init()

# custom modifiers
def DeactivatePBCsModifier(frame, data): # input needs to be in this format
    data.cell_.pbc = (False, False, False)

def DeleteIncompleteMoleculesModifier(frame, data):
    molIDs = np.unique(data.particles['Molecule Identifier'][...])
    for molID in molIDs:
        if np.sum(data.particles['Molecule Identifier'] == molID) != 3:
            om.ExpressionSelectionModifier(
                expression=f"Molecule Identifier == {molID}").modify(frame, data)
            om.DeleteSelectedModifier().modify(frame, data)

def AnalyseMoleculesModifier(frame, data):
    molIDs = np.unique(data.particles['Molecule Identifier'][...])
    nmol = len(molIDs)
    masses = data.particles['Mass'][...]
    positions = data.particles['Position'][...]
    table = od.DataTable()
    table.create_property('Molecule Identifier', data=molIDs)

    # internal properties
    sis = np.zeros((nmol,3))
    sjs = np.zeros((nmol,3))
    thetas = np.zeros(nmol)
    dipoles = np.zeros((nmol,3))
    for mol, molID in enumerate(molIDs):
        mask_O = (data.particles['Molecule Identifier'][...] == molID) & (data.particles['Particle Type'][...] == 1)
        mask_H = (data.particles['Molecule Identifier'][...] == molID) & (data.particles['Particle Type'][...] == 2)
        o = np.where(mask_O)[0][0]
        i = np.where(mask_H)[0][0]
        j = np.where(mask_H)[0][1]
        si = positions[o] - positions[i]
        sj = positions[o] - positions[j]
        sis[mol] = si
        sjs[mol] = sj
        thetas[mol] = math.acos(np.dot(si, sj) / (np.linalg.norm(si) * np.linalg.norm(sj)))
        dipoles[mol] = -(si + sj) / 2
    table.create_property('Si', data=np.linalg.norm(sis, axis=1))
    table.create_property('Sj', data=np.linalg.norm(sjs, axis=1))
    table.create_property('Theta', data=thetas)
    table.create_property('Si Vector', data=sis)
    table.create_property('Sj Vector', data=sjs)
    table.create_property('Dipole Vector', data=dipoles)

    # external
    com_droplet = data.tables['clusters']['Center of Mass'][...][0]
    coms = np.zeros((nmol, 3))
    for mol, molID in enumerate(molIDs):
        mask = data.particles['Molecule Identifier'][...] == molID
        pos = positions[mask]
        mass = masses[mask]
        com = np.sum(pos * mass[:, None], axis=0) / np.sum(mass)
        coms[mol] = com - com_droplet
    table.create_property('COM Vector', data=coms)
    table.create_property('COM Distance', data=np.linalg.norm(coms, axis=1))

    data.tables['molecules'] = table


def make_pipeline_droplet(filename, cutoff):
    pipeline = import_file(filename)
    pipeline.modifiers.append(DeactivatePBCsModifier)
    pipeline.modifiers.append(om.ClusterAnalysisModifier(cutoff=cutoff,
                                                         sort_by_size=True,
                                                         unwrap_particles=True,
                                                         compute_com=True,
                                                         compute_gyration=True))
    pipeline.modifiers.append(om.ExpressionSelectionModifier(expression="Cluster != 1"))
    pipeline.modifiers.append(om.DeleteSelectedModifier())
    pipeline.modifiers.append(DeleteIncompleteMoleculesModifier)
    pipeline.modifiers.append(AnalyseMoleculesModifier)
    return pipeline


def calculate_rho(pipeline, step=False):
    if step:
        steps = [step]
    else:
        steps = np.arange(0, pipeline.num_frames)
    hist = np.zeros(int(settings.rmax_hist / settings.dr_hist)+1)
    for step in tqdm(steps, desc="Calculating rho"):
        data = pipeline.compute(step)
        dist = data.tables['molecules']['COM Distance'][...]
        for i in range(len(dist)):
            r = dist[i]
            if r < settings.rmax_hist:
                bin_n = int(r/settings.dr_hist)
                hist[bin_n] += 1

    volumes = np.zeros(int(settings.rmax_hist / settings.dr_hist)+1)
    dist = np.arange(0, settings.rmax_hist, settings.dr_hist)
    for r in dist:
        volumes[i] = 4/3*np.pi * ((r + settings.dr_hist)**3 - r**3)

    return dist, hist/volumes


def calculate_dipole_projections(pipeline, step = False):
    if step:
        steps = [step]
    else:
        steps = np.arange(0, pipeline.num_frames)
    # steps constant, but nmol in droplet changes -> dtype 0 object
    projections = np.empty((len(steps),), dtype=object)
    for step in tqdm(steps, desc="Calculating dipole projection"):
        data = pipeline.compute(step)
        directions = data.tables['molecules']['COM Vector'][...]
        distances = data.tables['molecules']['COM Distance'][...]
        unit_directions = directions / distances
        dipoles = data.tables['molecules']['Dipole'][...]
        projections[step] = np.array([np.dot(udir, p)
                                      for udir, p in zip(unit_directions, dipoles)])
    return projections


def calculate_dipole_projections_by_distance(pipeline, step = False):
    if step:
        steps = [step]
    else:
        steps = np.arange(0, pipeline.num_frames)
    hist = np.zeros(int(settings.rmax_hist / settings.dr_hist)+1)
    hist_count = np.zeros(int(settings.rmax_hist / settings.dr_hist)+1)
    for step in tqdm(steps, desc="Calculating dipole projection by distance"):
        data = pipeline.compute(step)
        directions = data.tables['molecules']['COM Vector'][...]
        distances = data.tables['molecules']['COM Distance'][...]
        unit_directions = directions / distances
        dipoles = data.tables['molecules']['Dipole'][...]
        proj = np.array([np.dot(udir, p) for udir, p in zip(unit_directions, dipoles)])
        for i in range(len(distances)):
            r = distances[i]
            if r < settings.rmax_hist:
                bin_n = int(r/settings.dr_hist)
                hist[bin_n] += proj[i]
                hist_count[bin_n] += 1
    dist = np.arange(0, settings.rmax_hist, settings.dr_hist)
    return dist, hist/hist_count


def ExportMoleculePropertiesModifier(frame, data):
    molIDs = data.tables['molecules']['Molecule Identifier'][...]
    data.particles_.positions_[...] = data.particles.positions[...] - data.tables['clusters']['Center of Mass'][...][0]
    data.particles_.velocities_[...] = np.zeros((len(molIDs)*3,3))
    data.particles_.forces_[...] = np.zeros((len(molIDs)*3,3))
    for i, molID in enumerate(molIDs):
        mask_O = (data.particles['Molecule Identifier'][...] == molID) & (data.particles['Particle Type'][...] == 1)
        idx_O = np.where(mask_O)[0][0]
        data.particles_.velocities_[idx_O] = data.tables['molecules']['COM Vector'][i]
        data.particles_.forces_[idx_O] = data.tables['molecules']['Dipole Vector'][i]





# file = '../output/part_3_save/traj_eq.txt'
# pipeline_droplet = make_pipeline_droplet(file, 0.8)
# pipeline_droplet.modifiers.append(ExportMoleculePropertiesModifier)
# # print(pipeline_droplet.num_frames)
# data = pipeline_droplet.compute(6052)
# # export all properties
# from ovito.io import export_file
# export_file(pipeline_droplet, '../output/part_3_save/traj_eq_pipelined.*.txt',
#             format='lammps/dump', multiple_frames=True, every_nth_frame=10,
#             columns=['Particle Identifier', 'Molecule Identifier', 'Particle Type', 'Mass',
#                      'Position.X', 'Position.Y', 'Position.Z',
#                      'Velocity.X', 'Velocity.Y', 'Velocity.Z',
#                      'Force.X', 'Force.Y', 'Force.Z',])