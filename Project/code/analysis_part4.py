import matplotlib.pyplot as plt
import os
import numpy as np
import postprocessing
from postprocessing import PostprocessingTools
import settings
settings.init()








# ausrichtung by distance
# overall asphericity
# molecules in droplet over time -> freeze
# droplet diameter scaling with nparticles

trajfileeq = os.path.join(settings.path, f"part_3/traj_eq.txt")
trajfile = os.path.join(settings.path, f"part_3/traj.txt")
# pipeline = postprocessing.make_pipeline_droplet(trajfile, settings.cutoff)
# r, rho = postprocessing.calculate_rho(pipeline, 1)
# eq = PostprocessingTools(trajfileeq, 1001)
prod = PostprocessingTools(trajfile, every_nth_frame=10)
r, rho = prod.calculate_rho(rmax_hist=1.7)
# print(prod.data[0].tables['molecules']['COM Distance'][...])
# eq.export_dump_files()
plt.plot(r, rho)
plt.show()