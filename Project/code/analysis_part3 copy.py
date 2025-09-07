import matplotlib.pyplot as plt
import os
import numpy as np
from postprocessing import PostprocessingTools
import settings
settings.init()


folder = "part_3"
every_nth_frame_post = 10
smooth_hist = 3
cutoff = settings.cutoff


# ------------- Energies moving into equilibrium ------------- #
energyfileeq = os.path.join(settings.path, f"{folder}/energy_eq.txt")
energyfile = os.path.join(settings.path, f"{folder}/energy.txt")
energieseq = np.loadtxt(energyfileeq).T
energies = np.loadtxt(energyfile).T

timeeq = energieseq[0] / 1e3
epoteq = np.sum(energieseq[1:5], axis=0)
epotintraeq = np.sum(energieseq[3:5], axis=0)
epotintereq = np.sum(energieseq[1:3], axis=0)
ekineq = energieseq[5]
etoteq = np.sum(energieseq[1:], axis=0)

plt.plot(timeeq, epotintraeq, label=r'$E_{pot, intra}$', color='xkcd:orange')
plt.plot(timeeq, epotintereq, label=r'$E_{pot, inter}$', color='red')
plt.plot(timeeq, ekineq, label=r'$E_{kin}$', color='blue')
plt.plot(timeeq, etoteq, label=r'$E_{tot}$', color='black')
plt.xlabel(r"$t$ [ps]")
plt.ylabel(r"$E$ [kcal/mole]")
# plt.legend()
plt.savefig(os.path.join(settings.path, f"images/energies.png"), dpi=300, bbox_inches='tight', transparent=True)
plt.show()


plt.plot(timeeq, np.ones(1500)*epotintraeq[0], label=r'$E_{pot, intra}$', color='xkcd:orange')
plt.plot(timeeq, np.concatenate((np.ones(1200)*epotintereq[0],np.ones(300)*np.min(epotintereq))), label=r'$E_{pot, inter}$', color='red')
plt.plot(timeeq, np.concatenate((np.ones(1200)*ekineq[0],np.ones(300)*np.max(ekineq))), label=r'$E_{kin}$', color='blue')
plt.plot(timeeq, np.ones(1500)*etoteq[0], label=r'$E_{tot}$', color='black')
plt.xlabel(r"$t$ [ps]")
plt.ylabel(r"$E$ [kcal/mole]")
plt.legend(loc='lower center')
plt.savefig(os.path.join(settings.path, f"images/energies_start.png"), dpi=300, bbox_inches='tight', transparent=True)
plt.show()


def CostumModifier():
    return None


from ovito.io import import_file

pipeline = import_file("path/to/dump_file")


import ovito.modifiers as om

pipeline.modifiers.append(
    om.ClusterAnalysisModifier(
        cutoff=settings.cutoff,
        compute_com=True)
    )
pipeline.modifiers.append(
    CostumModifier()
    )


computed_data = pipeline.compute(step=1)

