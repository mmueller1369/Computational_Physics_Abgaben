import matplotlib.pyplot as plt
import os
import numpy as np
import math
import postprocessing
import settings
# import settings_SI as settings
from scipy.optimize import curve_fit
settings.init()

# ------------- Energies moving into equilibrium ------------- #
energyfileeq = os.path.join(settings.path, f"part_3/energy_eq.txt")
energyfile = os.path.join(settings.path, f"part_3/energy.txt")
energieseq = np.loadtxt(energyfileeq).T
energies = np.loadtxt(energyfile).T

timeeq = energieseq[0]*0.5 / 1e3
epoteq = np.sum(energieseq[1:5], axis=0)
epotintraeq = np.sum(energieseq[3:5], axis=0)
epotintereq = np.sum(energieseq[1:3], axis=0)
ekineq = energieseq[5]
etoteq = np.sum(energieseq[1:], axis=0)

plt.plot(timeeq, epotintraeq, label=r'$E_{pot, intra}$', color='xkcd:orange')
plt.plot(timeeq, epotintereq, label=r'$E_{pot, inter}$', color='red')
# plt.plot(timeeq, epoteq, label=r'$E_{pot, tot}$')
plt.plot(timeeq, ekineq, label=r'$E_{kin}$', color='blue')
plt.plot(timeeq, etoteq, label=r'$E_{tot}$', color='black')
plt.xlabel(r"$t$ [ps]")
plt.ylabel(r"$E$ [kcal/mole]")
plt.legend()
plt.show()


time = energies[0]*0.5 / 1e3 + timeeq[-1]
epot = np.sum(energies[1:5], axis=0)
epotintra = np.sum(energies[3:5], axis=0)
epotinter = np.sum(energies[1:3], axis=0)
ekin = energies[5]
etot = np.sum(energies[1:], axis=0)

plt.plot(timeeq, epotintraeq, label=r'$E_{pot, intra}$', color='xkcd:orange')
plt.plot(time, epotintra, color='xkcd:orange')
plt.plot(timeeq, epotintereq, label=r'$E_{pot, inter}$', color='red')
plt.plot(time, epotinter, color='red')
# plt.plot(timeeq, epoteq, label=r'$E_{pot, tot}$')
# plt.plot(time, epot)
plt.plot(timeeq, ekineq, label=r'$E_{kin}$', color='blue')
plt.plot(time, ekin, color='blue')
plt.plot(timeeq, etoteq, label=r'$E_{tot}$', color='black')
plt.plot(time, etot, color='black')
plt.axvline(timeeq[-1], ls='dashed', color='black', label=r'$t_{eq}$')
plt.xlabel(r"$t$ [ps]")
plt.ylabel(r"$E$ [kcal/mole]")
plt.legend()
plt.show()



# ------------- Temperature moving into equilibrium ------------- #
tempfileeq = os.path.join(settings.path, f"part_3/temp_eq.txt")
tempfile = os.path.join(settings.path, f"part_3/temp.txt")
tempfileeq = np.loadtxt(tempfileeq).T
tempfile = np.loadtxt(tempfile).T

plt.plot(timeeq, tempfileeq[1], label = r"$T$", color='blue')
plt.plot(time, tempfile[1], color='blue')
plt.axvline(timeeq[-1], ls='dashed', color='black', label=r'$t_{eq}$')
plt.xlabel(r"$t$ [ps]")
plt.ylabel(r"$T$ [K]")
plt.legend()
plt.show()



# ------------- Droplet radius moving into equilibrium ------------- #
trajfileeq = os.path.join(settings.path, f"part_3/traj_eq.txt")
trajfile = os.path.join(settings.path, f"part_3/traj.txt")
pipelineeq = postprocessing.make_pipeline_droplet(trajfileeq, settings.cutoff)
pipeline = postprocessing.make_pipeline_droplet(trajfile, settings.cutoff)
stepeq, radiieq, asphericitieseq = postprocessing.calculate_droplet_properties(pipelineeq)
step, radii, asphericities = postprocessing.calculate_droplet_properties(pipeline)

plt.plot(stepeq, radiieq, label = r"$R$", color='blue')
plt.plot(step, radii, color='blue')
# plt.axvline(stepeq[-1], ls='dashed', color='black', label=r'$t_{eq}$')
plt.xlabel(r"step")
plt.ylabel(r"$R_g$ [nm]")
plt.legend()
plt.show()

plt.plot(stepeq, asphericitieseq, label = r"$b$", color='blue')
plt.plot(step, asphericities, color='blue')
# plt.axvline(stepeq[-1], ls='dashed', color='black', label=r'$t_{eq}$')
plt.xlabel(r"$t$ [ps]")
plt.ylabel(r"$b$")
plt.legend()
plt.show()