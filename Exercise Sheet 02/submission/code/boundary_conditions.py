import numpy as np
import params


def periodic_boundary_conditions(
    data: np.ndarray, box_bounds: tuple, timestep: int
) -> np.ndarray:
    """
    Apply periodic boundary conditions to the simulation data.

    Parameters:
        data (numpy.ndarray): The simulation data.
        box_bounds (tuple): The bounds of the simulation box.
        timestep (int): The current timestep.

    Returns:
        numpy.ndarray: The updated simulation data with periodic boundary conditions applied.
    """
    configuration = data[timestep]

    for i, direction in enumerate(["x", "y", "z"]):
        index = params.properties[direction]
        lower_bound, upper_bound = box_bounds[i]
        length = upper_bound - lower_bound
        configuration[index, :] = np.where(
            (configuration[index, :] < lower_bound)
            | (configuration[index, :] > upper_bound),
            configuration[index, :]
            - length * np.floor(configuration[index, :] / length),
            configuration[index, :],
        )

    return configuration
