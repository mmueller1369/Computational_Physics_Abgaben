from matplotlib import pyplot as plt
import os
import numpy as np
import settings

settings.init()

part_a = np.loadtxt(os.path.join(settings.path, "temp_a.txt"))
time = part_a[:, 0]
temp = part_a[:, 1]

plt.plot(time, temp)
plt.show()
