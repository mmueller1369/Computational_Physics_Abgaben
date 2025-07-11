import matplotlib.pyplot as plt
import os
import numpy as np
import math
import postprocessing
# import settings
import settings_SI as settings
from scipy.optimize import curve_fit
settings.init()



# plt.plot(step, theta[:,0]*180/np.pi, label="angle")
# plt.show()

# plt.plot(step, si[:,0], label="si")
# plt.plot(step, sj[:,0], label="sj")
# plt.legend()
# plt.show()

energyfileeq = os.path.join(settings.path, f"part_3_save/energy_eq.txt")
energyfile = os.path.join(settings.path, f"part_3_save/energy.txt")
energieseq = np.loadtxt(energyfile).T
energies = np.loadtxt(energyfile).T
timeeq = energieseq[0]*0.5
time = energies[0]*0.5 + timeeq[-1]

plt.plot(timeeq, np.sum(energieseq[1:5], axis=0), label = r"$E_{pot}$", color='blue')
plt.plot(timeeq, energieseq[5], label = r"$E_{kin}$", color='blue')
# plt.plot(timeeq, np.sum(energieseq[1:], axis=0), label = r"$E_{tot}$", color='blue')
plt.plot(time, np.sum(energies[1:5], axis=0), label = r"$E_{pot}$", color='red')
plt.plot(time, energies[5], label = r"$E_{kin}$", color='red')
# plt.plot(time, np.sum(energies[1:], axis=0), label = r"$E_{tot}$", color='red')
plt.xlabel(r"$t$ [fs]")
plt.ylabel(r"$E$ [kcal/mole]")
plt.legend()
plt.show()

tempfileeq = os.path.join(settings.path, f"part_3_save/temp_eq.txt")
tempfile = os.path.join(settings.path, f"part_3_save/temp.txt")
tempfileeq = np.loadtxt(tempfileeq).T
tempfile = np.loadtxt(tempfile).T
timeeq = tempfileeq[0]*0.5
time = tempfile[0]*0.5 + timeeq[-1]
plt.plot(timeeq, tempfileeq[1], label = r"$T$", color='blue')
plt.plot(time, tempfile[1], label = r"$T$", color='red')
plt.xlabel(r"$t$ [fs]")
plt.ylabel(r"$T$ [K]")
plt.legend()
plt.show()