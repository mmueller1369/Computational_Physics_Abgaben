import numpy as np


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

    for dim in range(3):
        lower_bound, upper_bound = box_bounds[dim]
        length = upper_bound - lower_bound
        configuration[dim] = np.where(
            configuration[dim] < lower_bound,
            configuration[dim] + length * np.floor(configuration[dim] / length),
            np.where(
                configuration[dim] > upper_bound,
                configuration[dim] - length * np.floor(configuration[dim] / length),
                configuration[dim],
            ),
        )

    return configuration
