import execute
import settings
import initialize
import numpy as np

settings.init()
x, y, z, vx, vy, vz = initialize.InitializeAtoms()
f_initial = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3))
initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]

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
