import execute
import settings
import initialize
import numpy as np
import matplotlib.pyplot as plt

settings.init()
x, y, z, vx, vy, vz = initialize.InitializeAtoms()
f_initial = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]

"""
# ----------------- Part a ----------------- #
def B2(beta, eps, sigma, cutoff):
    def potentialLJ(r, eps, sigma, cutoff):
        sf6a = (sigma / cutoff) ** 6
        epotcut = 4.0 * eps * sf6a * (sf6a - 1.0)
        mask = (r < cutoff) & (r > 0)
        sf6 = np.zeros_like(r)
        sf6[mask] = (sigma / r[mask]) ** 6
        epot = np.zeros_like(r)
        epot[mask] = 4.0 * eps * sf6[mask] * (sf6[mask] - 1.0) - epotcut
        return epot

    deltar = cutoff / 1000
    r = np.arange(0, cutoff, deltar)
    U = potentialLJ(r, eps, sigma, cutoff)
    integral_core = np.exp(-beta * U) - 1
    integral_core *= 4 * np.pi * r**2  # conversion into spherical coordinates
    integral = -1 / 2 * np.sum(integral_core) * deltar
    return integral


beta = 1 / (settings.Tdesired * settings.kb)
sigma, cutoff = settings.sigma, 2.5 * settings.sigma
print(
    "For our parameter set: B2/eps =",
    B2(
        beta,
        1,  # set to unity
        sigma,
        cutoff,
    ),
)

epss = np.linspace(0, 1, 1000) / beta
B2_eps = [B2(beta, eps, sigma, cutoff) / sigma**3 for eps in epss]

plt.figure()
plt.plot(epss * beta, B2_eps)
plt.xlabel(r"$\epsilon\beta$")
plt.ylabel(r"$B_2/\sigma^3$")
plt.grid(True)
plt.show()
"""


# ----------------- Part b ----------------- #
equilibrated_config = execute.run_simulation(
    initial_config=initial_config,
    integrator="VelocityVerlet",
    force="LJ",
    steps=settings.nsteps_equi,
    thermostat="andersen_thermostat",
    thermostat_params=[settings.Tdesired, settings.nu, settings.deltat],
    n_thermostat=1,
    trajfile=False,
    tempfile=False,
    energyfile=False,
    pressfile=False,
    rdffile=False,
    n_save=10,
    simulation_name="Equilibration part b",
)

final_config = execute.run_simulation(
    initial_config=equilibrated_config,
    integrator="VelocityVerlet",
    force="LJ",
    steps=settings.nsteps_production,
    thermostat="andersen_thermostat",
    thermostat_params=[settings.Tdesired, settings.nu, settings.deltat],
    n_thermostat=1,
    trajfile=False,
    tempfile="b_temp",
    energyfile=False,
    pressfile="b_press",
    rdffile="b_gr",
    n_save=10,
    simulation_name="Production part b",
)
"""
# ----------------- Part a ----------------- #
settings.tau = 5000 * settings.deltat
settings.Tdesired = 300
settings.eps = 0.5 * settings.kb * settings.Tdesired
config_1 = execute.run_simulation(
    initial_config=initial_config,
    integrator="VelocityVerlet",
    force="LJ",
    steps=40000,
    thermostat="berendsen_thermostat",
    thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
    n_thermostat=1,
    trajfile=False,
    tempfile="temp_a_1",
    energyfile=False,
    rdffile=False,
    n_save=10,
    simulation_name="T=300K",
)

settings.Tdesired = 100
settings.eps = 0.5 * settings.kb * settings.Tdesired
config_2 = execute.run_simulation(
    initial_config=config_1,
    integrator="VelocityVerlet",
    force="LJ",
    steps=40000,
    thermostat="berendsen_thermostat",
    thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
    n_thermostat=1,
    trajfile=False,
    tempfile="temp_a_2",
    energyfile=False,
    rdffile=False,
    n_save=10,
    simulation_name="T=100K",
)


# ----------------- Part b ----------------- #
settings.tau = 5000 * settings.deltat
settings.Tdesired = 300
settings.eps = 0.5 * settings.kb * settings.Tdesired
config_1 = execute.run_simulation(
    initial_config=initial_config,
    integrator="VelocityVerlet",
    force="LJ",
    steps=40000,
    thermostat="andersen_thermostat",
    thermostat_params=[settings.Tdesired, settings.nu, settings.deltat],
    n_thermostat=1,
    trajfile=False,
    tempfile="temp_b_1",
    energyfile=False,
    rdffile=False,
    n_save=10,
    simulation_name="T=300K",
)

settings.Tdesired = 100
settings.eps = 0.5 * settings.kb * settings.Tdesired
config_2 = execute.run_simulation(
    initial_config=config_1,
    integrator="VelocityVerlet",
    force="LJ",
    steps=40000,
    thermostat="andersen_thermostat",
    thermostat_params=[settings.Tdesired, settings.nu, settings.deltat],
    n_thermostat=1,
    trajfile=False,
    tempfile="temp_b_2",
    energyfile=False,
    rdffile=False,
    n_save=10,
    simulation_name="T=100K",
)


# ----------------- Part c+d ----------------- #
settings.tau = 500 * settings.deltat
settings.Tdesired = 300
settings.eps = 0.5 * settings.kb * settings.Tdesired
equilibrated_config = execute.run_simulation(
    initial_config=initial_config,
    integrator="VelocityVerlet",
    force="LJ",
    steps=40000,
    thermostat="berendsen_thermostat",
    thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
    n_thermostat=1,
    trajfile=False,
    tempfile=False,
    energyfile=False,
    rdffile=False,
    n_save=10,
    simulation_name="Equilibration",
)

final_config = execute.run_simulation(
    initial_config=equilibrated_config,
    integrator="VelocityVerlet",
    force="LJ",
    steps=40000,
    thermostat="berendsen_thermostat",
    thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
    n_thermostat=1,
    trajfile=False,
    tempfile="temp_cd",
    energyfile="energy_cd",
    rdffile="rdf_cd",
    n_save=10,
    simulation_name="Production",
)

# ----------------- Part e ----------------- #
settings.tau = 500 * settings.deltat
settings.Tdesired = 300
settings.eps = 0.5 * settings.kb * settings.Tdesired
equilibrated_config = execute.run_simulation(
    initial_config=initial_config,
    integrator="VelocityVerlet",
    force="LJ",
    steps=40000,
    thermostat="andersen_thermostat",
    thermostat_params=[settings.Tdesired, settings.nu, settings.deltat],
    n_thermostat=1,
    trajfile=False,
    tempfile=False,
    energyfile=False,
    rdffile=False,
    n_save=10,
    simulation_name="Equilibration",
)

final_config = execute.run_simulation(
    initial_config=equilibrated_config,
    integrator="VelocityVerlet",
    force="LJ",
    steps=40000,
    thermostat="andersen_thermostat",
    thermostat_params=[settings.Tdesired, settings.nu, settings.deltat],
    n_thermostat=1,
    trajfile=False,
    tempfile="temp_e",
    energyfile="energy_e",
    rdffile="rdf_e",
    n_save=10,
    simulation_name="Production",
)
"""
