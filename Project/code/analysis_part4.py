import matplotlib.pyplot as plt
import os
import postprocessing
import settings
settings.init()

trajfileeq = os.path.join(settings.path, f"part_3_save/traj_eq.txt")
trajfile = os.path.join(settings.path, f"part_3_save/traj.txt")

# ausrichtung by distance
# overall asphericity
# molecules in droplet over time -> freeze

pipeline = postprocessing.make_pipeline_droplet(trajfile, settings.cutoff)
r, rho = postprocessing.calculate_rho(pipeline)
plt.plot(r, rho)
plt.show()