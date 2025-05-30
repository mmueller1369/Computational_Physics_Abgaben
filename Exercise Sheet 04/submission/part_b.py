"""
Teil b des Exercise Sheet 4
ausführen nach main.py
"""

import os
import numpy as np
import settings
import statistical_error as se
import g_r

settings.init()
# histogramme über gesamte trajektorie
histogramme_gesamt = np.loadtxt(os.path.join(settings.path, "histogram_z_a.txt"))
# unterteile in blocks
block_hist = np.array_split(histogramme_gesamt, settings.nblocks)
adsorption_values = []

for i, hist in enumerate(block_hist):
    rho_z = g_r.calc_density_1d(hist)
    # g_r.plot_density_1d(rho_z, "z")
    adsorption = se.calc_adsorption(rho_z)
    print(f"Adsorption: {adsorption}")
    adsorption_values.append(adsorption)
# calculate statistical error
error = se.calc_statistical_error(adsorption_values)
print(f"Statistical error: {error}")
