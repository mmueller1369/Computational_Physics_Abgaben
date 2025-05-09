import numpy as np
import params


def velocity_rescaling_thermostat(
    data: np.ndarray,
    timestep: int,
    dt_thermostat: int,
    T: float,
) -> np.ndarray:
    """
    Apply a velocity rescaling thermostat to the simulation data.

    Parameters:
        data (numpy.ndarray): The simulation data.
        timestep (int): The current timestep.
        dt_thermostat (int): The time step for thermostat updates.
        T (float): The target temperature.

    Returns:
        numpy.ndarray: The updated simulation data with the thermostat applied.
    """
    configuration = data[timestep]
    nparticles = configuration.shape[1]

    if timestep % dt_thermostat == 0:
        # Calculate the current temperature
        velocities = configuration[
            params.properties["vx"] : params.properties["vz"] + 1, :
        ]
        masses = configuration[params.properties["mass"], :]
        kinetic_energy = 0.5 * np.sum(masses * np.sum(velocities**2, axis=0))
        current_temperature = (2 * kinetic_energy) / (3 * nparticles * params.k_B)

        # Calculate the scaling factor
        scaling_factor = np.sqrt(T / current_temperature)

        # Rescale velocities
        configuration[
            params.properties["vx"] : params.properties["vz"] + 1, :
        ] *= scaling_factor

    return configuration
