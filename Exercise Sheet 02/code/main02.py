# from simulation_tools import md_simulation
from data_io import export_data
from part_a import lj_system_initial_configuration, create_geometry
import numpy as np


data = lj_system_initial_configuration
data = create_geometry(data)

export_data(
    np.array([data]),
    selected_properties=True,
    dt_export=1,
    filename="initial_test.dat",
    box_bounds=((0, 10), (0, 10), (0, 0)),
)
