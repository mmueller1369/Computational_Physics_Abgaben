import numpy as np
import os
from numba import njit, prange
from tqdm import tqdm


class MDSettings:
    """Verwaltet alle Simulationsparameter."""

    def __init__(self, n1, n2, n3, rho, deltat, kb, Tdesired, mass, eps, sigma, cutoff=None, eps_wall=None, sigma_wall=None, cutoff_wall=None):
        self.n1 = n1 # number of particles in each direction
        self.n2 = n2
        self.n3 = n3
        self.rho = rho # particle density N/V (in units of sigma^-3)
        self.deltat = deltat # time step (fs)
        self.kb = kb  # boltzmann's constant (kcal/mole/K)
        self.Tdesired = Tdesired  # temperature of the experiment in K
        self.mass = mass  # mass of the LJ particles (gram/mole)
        self.eps = eps# eps in LJ (kcal/mole)
        self.sigma = sigma # sigma in LJ (nm)
        self.cutoff = cutoff * self.sigma # cutoff (in units of sigma)
        self.eps_wall = eps_wall # eps for wall (kcal/mole)
        self.sigma_wall = sigma_wall # sigma for wall (nm)
        self.cutoff_wall = cutoff_wall * self.sigma_wall # cutoff for wall (in units of sigma_wall)
        self.internal_params()

    def internal_params(self):
        self.nparticles = self.n1 * self.n2 * self.n3 #total number of particles
        self.lx = self.n1 / (self.rho ** (1 / 3)) # box lenghts in each direction
        self.ly = self.n2 / (self.rho ** (1 / 3))
        self.lz = self.n3 / (self.rho ** (1 / 3))
        self.xlo = 0 * self.sigma # box limits in each direction
        self.xhi = self.lx * self.sigma
        self.ylo = 0 * self.sigma
        self.yhi = self.ly * self.sigma
        self.zlo = 0 * self.sigma
        self.zhi = self.lz * self.sigma
        self.convtemp = 238845.9 # from (gram/mole)*(nm/fs)^2/((kcal/mole)/K) to K
        self.convvelocity = 4.1868e5
        self.ouput_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output") # path for the output
        if not os.path.exists(self.ouput_path): # creates ouput folder if it doesn't exist yet
            os.makedirs(self.ouput_path)

    def change_param(self, param, param_value):
        """
        Setzt den Parameter mit Namen param (als String) auf param_value
        und berechnet anschließend die abgeleiteten Größen neu.
        Internal-Parameter (abgeleitete Größen) dürfen nicht direkt gesetzt werden.
        """
        # internal_params = [
        #     'nparticles', 'lx', 'ly', 'lz', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi', 'ouput_path', 'deltaxyz'
        # ]
        # if param in internal_params:
        #     raise AttributeError(f"'{param}' is an internal parameter and can't be changed.")
        # if not hasattr(self, param):
        #     raise AttributeError(f"Parameter '{param}' doesn't exist.")
        setattr(self, param, param_value)
        self.internal_params()

# self.thermostat = 2
#         self.tau = 500 * self.deltat
#         self.n_thermostat = 1
#         self.nu = 0.01
#         self.deltar = 0.05 * self.sigma
#         self.rmax = 1 / 2 * max(self.lx, self.ly, self.lz) * self.sigma
#         self.n_analyze = 10
#         self.n_gr = int(self.nsteps_production / self.n_analyze)
#         self.nblocks = 6

#         self.n_save = 10

class MDInitializer:
        
    """Initialisiert die Atome und Geschwindigkeiten."""

    def __init__(self, settings: MDSettings):
        self.settings = settings
        self.deltaxyz = (self.settings.xhi - self.settings.xlo) / self.settings.n1

    def initialize_lattice(self):
        import random
        sett = self.settings
        n = sett.nparticles
        x = np.zeros(n)
        y = np.zeros(n)
        z = np.zeros(n)
        vx = np.zeros(n)
        vy = np.zeros(n)
        vz = np.zeros(n)
        n, nx, ny, nz = 0, 0, 0, 0
        pbar = tqdm(total=n, desc="Initialization")
        while nx < sett.n1:
            ny = 0
            while ny < sett.n2:
                nz = 0
                while nz < sett.n3:
                    x[n] = nx * self.deltaxyz + self.deltaxyz / 2.0
                    y[n] = ny * self.deltaxyz + self.deltaxyz / 2.0
                    z[n] = nz * self.deltaxyz + self.deltaxyz / 2.0
                    vx[n] = 0.5 - random.randint(0, 1)
                    vy[n] = 0.5 - random.randint(0, 1)
                    vz[n] = 0.5 - random.randint(0, 1)
                    n += 1
                    nz += 1
                    pbar.update(1)
                ny += 1
            nx += 1
        pbar.close()
        # Linearer Impuls nullen
        vx -= np.mean(vx)
        vy -= np.mean(vy)
        vz -= np.mean(vz)
        # Temperatur reskalieren
        rescale_velocity = MDThermostat(self.settings).rescale_velocity()
        vx, vy, vz = rescale_velocity(vx, vy, vz, self.settings)
        return x, y, z, vx, vy, vz

    
class MDAnalyze:
    def __init__(self):
        pass

    def temperature(self, vx, vy, vz, settings: MDSettings):
        vsq = np.sum(vx**2 + vy**2 + vz**2)
        temp = settings.mass * vsq / 2.0 / settings.kb / settings.nparticles * settings.convtemp
        return temp
    
    @njit(parallel=True)
    def KineticEnergy(self):
        sett = self.settings
        vx, vy, vz, mass = sett.vx, sett.vy, sett.vz, sett.mass
        ekin = 0
        for i in prange(sett.nparticles):
            ekin += 0.5 * mass * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]) * self.convvelocity
        return ekin
    
class MDThermostat:
    def __init__(self, settings: MDSettings):
        self.settings = settings
        self.func_temperature = MDAnalyze

    def rescale_velocity(self, vx, vy, vz):
        T_current = MDAnalyze.temperature(vx, vy, vz)
        factor = np.sqrt(self.settings.T_desired / T_current)
        return vx * factor, vy * factor, vz * factor
    
    def berendsen_thermostat(self, vx, vy, vz, T1, T2, tau, dt):
        multiplier = np.sqrt(1 + (dt / tau) * ((T1 / T2) - 1))
        vx = vx * multiplier
        vy = vy * multiplier
        vz = vz * multiplier
        return vx, vy, vz

    def andersen_thermostat(self, vx, vy, vz, T0, Tsystem, nu, dt):
        variance = np.sqrt(self.settings.kb * T0 / self.settings.mass / self.convtemp)
        for i, _ in enumerate(vx):
            if np.random.rand() < nu * dt:
                vx[i] = np.random.normal(0, variance)
                vy[i] = np.random.normal(0, variance)
                vz[i] = np.random.normal(0, variance)
    return vx, vy, vz



class MDFORCES:
    """Berechnet Kräfte für verschiedene Potentiale."""

    def __init__(self, settings: MDSettings):
        self.settings = settings

    def LJ(self, x, y, z):
        n = self.settings.nparticles
        fx = np.zeros(n)
        fy = np.zeros(n)
        fz = np.zeros(n)
        epot = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                rijx = self.pbc(x[i], x[j], self.settings.xlo, self.settings.xhi)
                rijy = self.pbc(y[i], y[j], self.settings.ylo, self.settings.yhi)
                rijz = self.pbc(z[i], z[j], self.settings.zlo, self.settings.zhi)
                r2 = rijx**2 + rijy**2 + rijz**2
                sf2 = self.settings.sigma**2 / r2
                sf6 = sf2**3
                epot += 4.0 * self.settings.eps * sf6 * (sf6 - 1.0)
                ff = 24.0 * self.settings.eps * sf6 * (sf6 - 0.5) / r2
                fx[i] -= ff * rijx
                fy[i] -= ff * rijy
                fz[i] -= ff * rijz
                fx[j] += ff * rijx
                fy[j] += ff * rijy
                fz[j] += ff * rijz
        return fx, fy, fz, epot

    @staticmethod
    def pbc(xi, xj, xlo, xhi):
        l = xhi - xlo
        rij = (xj - xi) % l
        if rij > 0.5 * l:
            rij -= l
        return rij


class MDEngine:
    """Führt die Zeitintegration und das Simulationsmanagement durch."""

    def __init__(
        self, settings: MDSettings, forces: MDFORCES, initializer: MDInitializer
    ):
        self.settings = settings
        self.forces = forces
        self.initializer = initializer

    def velocity_verlet(self, x, y, z, vx, vy, vz, fx, fy, fz):
        N = len(x)
        dt = self.settings.deltat
        mass = self.settings.mass
        # Positionen updaten
        x += vx * dt + 0.5 * fx * dt**2 / mass
        y += vy * dt + 0.5 * fy * dt**2 / mass
        z += vz * dt + 0.5 * fz * dt**2 / mass
        # Kräfte neu berechnen
        fx_new, fy_new, fz_new, epot = self.forces.force_LJ(x, y, z)
        # Geschwindigkeiten updaten
        vx += 0.5 * (fx + fx_new) * dt / mass
        vy += 0.5 * (fy + fy_new) * dt / mass
        vz += 0.5 * (fz + fz_new) * dt / mass
        return x, y, z, vx, vy, vz, fx_new, fy_new, fz_new, epot

    def run(self, steps, thermostat=None, thermostat_params=None):
        # Initialisierung
        x, y, z, vx, vy, vz = self.initializer.initialize_atoms()
        fx, fy, fz, epot = self.forces.force_LJ(x, y, z)
        # Hauptschleife
        for step in range(steps):
            x, y, z, vx, vy, vz, fx, fy, fz, epot = self.velocity_verlet(
                x, y, z, vx, vy, vz, fx, fy, fz
            )
            # Thermostat ggf. anwenden (optional)
            # ...
        return x, y, z, vx, vy, vz, fx, fy, fz


class MDIO:
    """Kümmert sich um Dateioperationen (Energie, Trajektorie, Temperatur, Druck)."""

    def __init__(self, settings: MDSettings):
        self.settings = settings

    def write_energy(self, fileenergy, itime, epot, ekin, vx2, vy2, vz2):
        fileenergy.write(
            f"{itime} {epot:.6e} {ekin:.6e} {vx2:.6e} {vy2:.6e} {vz2:.6e}\n"
        )

    def write_temp(self, filetemp, itime, temp):
        filetemp.write(f"{itime} {temp:.6e}\n")

    def write_press(self, filepress, itime, press):
        filepress.write(f"{itime} {press:.6e}\n")

    def write_trajectory(self, fileoutput, itime, x, y, z, vx, vy, vz, fx, fy, fz):
        fileoutput.write("ITEM: TIMESTEP \n")
        fileoutput.write(f"{itime}\n")
        fileoutput.write("ITEM: NUMBER OF ATOMS \n")
        fileoutput.write(f"{self.settings.nparticles}\n")
        fileoutput.write("ITEM: BOX BOUNDS \n")
        fileoutput.write(f"{self.settings.xlo} {self.settings.xhi}\n")
        fileoutput.write(f"{self.settings.ylo} {self.settings.yhi}\n")
        fileoutput.write(f"{self.settings.zlo} {self.settings.zhi}\n")
        fileoutput.write("ITEM: ATOMS id type x y z vx vy vz fx fy fz\n")
        for i in range(len(x)):
            fileoutput.write(
                f"{i} {i} {x[i]:.6e} {y[i]:.6e} {z[i]:.6e} {vx[i]:.6e} {vy[i]:.6e} {vz[i]:.6e} {fx[i]:.6e} {fy[i]:.6e} {fz[i]:.6e}\n"
            )


class MDAnalysis:
    """Analyse-Tools wie g(r), Histogramme, Blockmittelung."""

    def __init__(self, settings: MDSettings):
        self.settings = settings

    def histogram(self, x, y, z, bin_width, rmax):
        n_bins = int(rmax / bin_width)
        hist = np.zeros(n_bins)
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                rijx = MDFORCES.pbc(x[i], x[j], self.settings.xlo, self.settings.xhi)
                rijy = MDFORCES.pbc(y[i], y[j], self.settings.ylo, self.settings.yhi)
                rijz = MDFORCES.pbc(z[i], z[j], self.settings.zlo, self.settings.zhi)
                r2 = rijx**2 + rijy**2 + rijz**2
                if r2 < rmax**2:
                    r = np.sqrt(r2)
                    bin_n = int(r / bin_width)
                    if bin_n < n_bins:
                        hist[bin_n] += 2
        return hist

    def block_averages(self, data, num_blocks):
        block_size = len(data) // num_blocks
        block_means = np.zeros(num_blocks)
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size if i < num_blocks - 1 else len(data)
            block_means[i] = np.mean(data[start:end])
        mean = np.mean(block_means)
        error = np.std(block_means, ddof=1) / np.sqrt(num_blocks)
        return block_means, mean, error


# ...Ende der Grundstruktur. Weitere Methoden können nach Bedarf ergänzt werden.
