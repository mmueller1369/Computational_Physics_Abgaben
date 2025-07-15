import math
from tqdm import tqdm
import numpy as np
from scipy.optimize import curve_fit
from ovito.io import import_file, export_file
import ovito.modifiers as om
import ovito.data as od
import forces
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
            mask = data.particles['Molecule Identifier'] == molID
            data.particles_.delete_elements(mask)

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

def MeasureHCOMModifier(frame, data):
    atIDs = data.particles['Particle Identifier'][...]
    data.particles_.create_property('HCOM Dist', dtype=np.float64, data=np.full(len(atIDs), np.nan))
    mask = data.particles['Particle Type'] == 2
    com_droplet = data.tables['clusters']['Center of Mass'][...][0]
    for at, atID in enumerate(atIDs[mask]):
        pos = data.particles['Position'][at]
        rel_pos = pos - com_droplet
        data.particles_['HCOM Dist'][at] = np.linalg.norm(rel_pos)

    

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


def fermi_distribution(r, r0, b, a):
    exponent = (r-r0)/a
    return b / (np.exp(exponent)+1)


class PostprocessingTools:
    def __init__(self, filename, every_nth_frame=1, cutoff=settings.cutoff):
        # initializing the pipeline
        self.filename = filename
        self.pipeline = import_file(filename)
        self.pipeline.modifiers.append(DeactivatePBCsModifier)
        self.pipeline.modifiers.append(om.ClusterAnalysisModifier(
            cutoff=cutoff,
            sort_by_size=True,
            unwrap_particles=True,
            compute_com=True,
            compute_gyration=True))
        self.pipeline.modifiers.append(om.ExpressionSelectionModifier(
            expression="Cluster != 1"))
        self.pipeline.modifiers.append(om.DeleteSelectedModifier())
        self.pipeline.modifiers.append(DeleteIncompleteMoleculesModifier)
        self.pipeline.modifiers.append(AnalyseMoleculesModifier)
        self.pipeline.modifiers.append(MeasureHCOMModifier)

        # calculating the data
        self.every_nth_frame = every_nth_frame
        self.ndata = int(self.pipeline.num_frames / every_nth_frame)
        self.data = np.empty((self.ndata,), dtype=object)
        pbar = tqdm(total=self.ndata, desc="Computing data from pipeline")
        for s in range(self.ndata):
            step = s*every_nth_frame
            self.data[s] = self.pipeline.compute(step)
            pbar.update(1)
        pbar.close()
    

    def _setup_steps(self, step):
        if step is not None:
            if isinstance(step, int):
                steps = np.array([step])
            else:
                steps = step
        else:
            steps = np.arange(0, self.ndata)
        return steps


    def calculate_rho(self, step=None, rmax_hist=settings.rmax_hist, dr_hist=settings.dr_hist, only_hatoms=False):
        steps = self._setup_steps(step)
        total_bins = int(rmax_hist / dr_hist) + 1
        hist = np.zeros(total_bins)
        for s in steps:
            if only_hatoms:
                dist = self.data[s].particles['HCOM Dist'][...]
            else:
                dist = self.data[s].tables['molecules']['COM Distance'][...]
            for i in range(len(dist)):
                r = dist[i]
                if r < rmax_hist:
                    bin_n = int(r/dr_hist)
                    hist[bin_n] += 1

        volumes = np.zeros(total_bins)
        for i in range(total_bins):
            r_inner = i * dr_hist
            r_outer = (i + 1) * dr_hist
            volumes[i] = 4/3*np.pi * (r_outer**3 - r_inner**3)

        dist = (np.arange(total_bins) + 0.5) * dr_hist
        rho = hist/volumes/len(steps)
        params, pcov = curve_fit(fermi_distribution, dist[3:], rho[3:], p0=[1.2, 35, 0.1])

        return dist, rho, params, pcov


    def calculate_dipole_projections(self, step=None):
        steps = self._setup_steps(step)
        # steps constant, but nmol in droplet changes -> dtype 0 object
        projections = np.empty((len(steps),), dtype=object)
        for s in steps:
            directions = self.data[s].tables['molecules']['COM Vector'][...]
            distances = self.data[s].tables['molecules']['COM Distance'][...]
            unit_directions = directions / distances[:, np.newaxis]
            dipoles = self.data[s].tables['molecules']['Dipole Vector'][...]
            projections[s] = np.array([np.dot(udir, p) 
                                       for udir, p in zip(unit_directions, dipoles)])
            
        return projections


    def calculate_dipole_projections_by_distance(self, step=None, rmax_hist=settings.rmax_hist, dr_hist=settings.dr_hist):
        steps = self._setup_steps(step)
        total_bins = int(rmax_hist / dr_hist) + 1
        hist = np.zeros(total_bins)
        hist_count = np.zeros(total_bins)
        projections = self.calculate_dipole_projections(step)
        for s in steps:
            projection = projections[s]
            distances = self.data[s].tables['molecules']['COM Distance'][...]
            for i in range(len(distances)):
                r = distances[i]
                if r < rmax_hist:
                    bin_n = int(r/dr_hist)
                    hist[bin_n] += projection[i]
                    hist_count[bin_n] += 1
        dist = np.linspace(0, rmax_hist, total_bins)
        return dist, hist/hist_count


    def calculate_droplet_properties(self, step=None, rmax_hist=settings.rmax_hist, dr_hist=settings.dr_hist, smooth_hist = 3):
        steps = self._setup_steps(step)
        rgs = np.zeros(len(steps))
        asphericities = np.zeros(len(steps))
        pbar = tqdm(total=len(steps), desc="Calculating droplet properties")
        params = np.zeros((len(steps), 3))
        pcovs = np.zeros((len(steps), 3, 3))
        for s in steps:
            # calculating properties obtained from the clustering
            gyration = self.data[s].tables['clusters']['Gyration Tensor'][0]
            matrix = np.array([[gyration[0], gyration[3], gyration[4]],
                            [gyration[3], gyration[1], gyration[5]],
                            [gyration[4], gyration[5], gyration[2]]])
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            rgs[s] = math.sqrt(np.sum(eigenvalues))
            asphericities[s] = 3/2*np.max(eigenvalues) - np.sum(eigenvalues)/2
            # fitting the density to a fermi distribution to obtain the radius
            try:
                # adding steps within smooth-hist range if possible
                steps_hist = np.arange(max(0, s-smooth_hist),
                                       min(len(steps), s+smooth_hist))
                _, _, param, pcov = self.calculate_rho(step=steps_hist,
                                                        rmax_hist=rmax_hist,
                                                        dr_hist=dr_hist)
                params[s] = param
                pcovs[s] = pcov
            except RuntimeError:
                params[s] = np.nan
                pcovs[s] = np.nan

            pbar.update(1)
        pbar.close()
        return steps, rgs, asphericities, params, pcovs
    

    def calculate_electrostatic_potential_and_field(self, step=None, rmax_hist=settings.rmax_hist, dr_hist=settings.dr_hist):
        steps = self._setup_steps(step)
        total_bins = int(rmax_hist / dr_hist) + 1
        charge_by_type = {1: settings.qO, 2: settings.qH}
        gamma_cut = forces.gamma(settings.cutoff, settings.alpha)
        energy = np.zeros(total_bins)
        field = np.zeros(total_bins)
        pbar = tqdm(total=len(steps), desc="Calculating electrostatic potential and field")
        for s in steps:
            positions = self.data[s].particles['Position'][...]
            types = self.data[s].particles['Particle Type'][...]
            for i in range(len(positions)):
                pos1 = positions[i]
                q1 = charge_by_type[types[i]]
                for j in range(i+1, len(positions)):
                    pos2 = positions[j]
                    q2 = charge_by_type[types[j]]
                    rol = np.linalg.norm(pos1 - pos2)
                    if rol < rmax_hist:
                        bin_n = int(rol/dr_hist)
                        if bin_n < total_bins:
                            ff_coul, e_coul = forces.ffe_coul(rol, q1, q2, settings.eps0_el, settings.alpha, settings.cutoff, gamma_cut)
                            energy[bin_n] += e_coul
                            field[bin_n] += ff_coul * rol
            pbar.update(1)

        dist = (np.arange(total_bins) + 0.5) * dr_hist
        return dist, energy/len(steps), field/len(steps)


    def export_dump_files(self, name=None):
        if name is None:
            name = self.filename[:-4] + '_pipelined.*.txt'
        self.pipeline.modifiers.append(ExportMoleculePropertiesModifier)
        export_file(self.pipeline, name,
                    format='lammps/dump', multiple_frames=True, every_nth_frame=self.every_nth_frame,
                    columns=['Particle Identifier', 'Molecule Identifier', 'Particle Type', 'Mass',
                             'Position.X', 'Position.Y', 'Position.Z',
                             'Velocity.X', 'Velocity.Y', 'Velocity.Z',
                             'Force.X', 'Force.Y', 'Force.Z',])