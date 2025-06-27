import execute
import settings
import initialize
import numpy as np

settings.init()

# ----------------- Part b ----------------- #
x, y, z, vx, vy, vz = initialize.InitializeAtomsBond()
f_initial = np.zeros(shape=(settings.n1 * settings.n2 * settings.n3 *2))
initial_config = [x, y, z, vx, vy, vz, f_initial, f_initial, f_initial]
equilibrated_config = execute.run_simulation_bonded(
    initial_config=initial_config,
    integrator="VelocityVerletBond",
    force="LJBond",
    steps=settings.nsteps_equi,
    thermostat="berendsen_thermostat",
    thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
    n_thermostat=1,
    trajfile1=False,
    trajfile2=False,
    energyfile=False,
    rdffile1=False,
    rdffile2=False,
    n_save1=1,
    n_save2=50,
    simulation_name="Equilibration",
)
final_config = execute.run_simulation_bonded(
    initial_config=equilibrated_config,
    integrator="VelocityVerletBond",
    force="LJBond",
    steps=settings.nsteps_production,
    thermostat="berendsen_thermostat",
    thermostat_params=[settings.Tdesired, settings.tau, settings.deltat],
    n_thermostat=1,
    trajfile1='TrajectoryforC',
    trajfile2='TrajectoryforD',
    energyfile='Energy',
    rdffile1='g(r)_1',
    rdffile2='g(r)_2',
    n_save1=1,
    n_save2=50,
    simulation_name="Production",
)