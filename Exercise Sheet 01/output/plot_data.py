"""
Author: Elias Jedam & Matthias Müller
Date: 02/05/2025
This file plot some data from the simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

planet_list = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto",
]


def plot3d_500years(simulation, positions):
    """_summary_

    Args:
        simulation (_type_): _description_
        positions (_type_): _description_
    """
    # Assuming positions.npy contains shape (num_planets, timesteps, 3)
    _, num_planets, _ = positions.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot each planet's trajectory
    for i in range(num_planets):
        x, y, z = positions[:, i, :].T  # Transpose to get x, y, z coordinates
        ax.plot(x, y, z, label=f"{planet_list[i]}")

    # Add labels and title
    ax.set_xlabel("X Position (AU)")
    ax.set_ylabel("Y Position (AU)")
    ax.set_zlabel("Z Position (AU)")
    ax.set_title(f"3D Trajectories of Planets over next 500 years ({simulation})")

    # Show legend
    ax.legend()

    # Show the plot
    plt.savefig(
        os.path.join(basis_pfad, "output", f"3d_trajectory_{simulation}_500years.png")
    )
    plt.show()


def plot_planet_for_years_2d(planet_index, positions, simulation, years=100 * 365 * 2):
    """_summary_

    Args:
        planet_index (_type_): _description_
        positions (_type_): _description_
        simulation (_type_): _description_
        years (_type_, optional): _description_. Defaults to 100*365*2.
    """
    # Example: positions.shape = (timesteps, planets, 3)
    # For planet index 3 (e.g. Mars)
    x = positions[:years, -1, 0]
    y = positions[:years, -1, 1]
    timesteps = np.arange(len(x))

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=timesteps, cmap="viridis", s=1)
    plt.xlabel("X Position (AU)")
    plt.ylabel("Y Position (AU)")
    plt.title(
        f"2D Trajectory of {planet_list[planet_index]} (Colored by Time {simulation})"
    )
    plt.colorbar(scatter, label="Timestep")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(
        os.path.join(
            basis_pfad,
            "output",
            f"2d_trajectory_{simulation}_{planet_list[planet_index]}100years.png",
        )
    )
    plt.show()


def plot3d_positions(
    positions,
    simulation,
    timestep=50 * 365 * 2,
):
    """_summary_

    Args:
        positions (_type_): _description_
        simulation (_type_): _description_
        timestep (_type_, optional): _description_. Defaults to 50*365*2.
    """
    # 3D plot for the positions of all planets at a single timestep
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x = positions[timestep, :, 0]
    y = positions[timestep, :, 1]
    z = positions[timestep, :, 2]

    scatter = ax.scatter(
        x, y, z, c=np.arange(len(x)), cmap="viridis", s=30
    )  # Increased size to 100
    ax.set_xlabel("X Position (AU)")
    ax.set_ylabel("Y Position (AU)")
    ax.set_zlabel("Z Position (AU)")
    ax.set_title(f"3D Positions of Planets at Timestep {timestep} ({simulation})")
    fig.colorbar(scatter, label="Planet Index")

    plt.savefig(
        os.path.join(
            basis_pfad, "output", f"3d_positions_timestep_{timestep}_{simulation}.png"
        )
    )
    plt.show()


# This function is not working as expected
# def calc_period(planet=9, mass=0.0127e-5 * 10**29, G=6.67408e-11):
#    radii = np.linalg.norm(positions[:, planet, :], axis=1)  # distances from the Sun
#    r_peri = np.min(radii)
#    r_aphel = np.max(radii)
#    a = (r_peri + r_aphel) / 2
#    T = 2 * np.pi * np.sqrt((a * 1.49597870691e11) ** 3 / (G * mass))
#    print(f"Estimated semi-major axis (AU): {a:.2f}")
#    print(f"Estimated period (years): {T / (365 * 24 * 3600):.2f}")
#    # why doesnt it work


def chat_gpt_calc_period(planet, positions):
    """_summary_

    Args:
        planet (_type_): _description_
        positions (_type_): _description_
    """

    # 1. Extract Pluto's position over time (index 9 if Pluto is the 10th planet)
    planet_pos = positions[:, planet, :]  # shape: (timesteps, 3)

    # 2. Compute radial distance from the Sun at each timestep
    radii = np.linalg.norm(planet_pos, axis=1)  # shape: (timesteps,)

    # 3. Find minima in the radial distance = perihelion passages
    # Since we want *minima*, we find peaks in the *negative* of the signal
    perihelion_indices, _ = find_peaks(-radii, distance=300)  # distance filters noise

    # 4. Calculate the time between perihelion passages
    timestep_interval = 1 / 2  # Assuming positions are sampled every 1 day
    if len(perihelion_indices) >= 2:
        periods = np.diff(perihelion_indices) * timestep_interval  # days
        avg_period_days = np.mean(periods)
        print(
            f"Estimated orbital period of {planet_list[planet]}: {avg_period_days / 365.25:.2f} years"
        )
    else:
        print("Not enough perihelion passages to estimate period.")


if __name__ == "__main__":
    print("Plotting data...")
    basis_pfad = os.path.dirname(os.path.dirname(__file__))
    for sim in ["euler", "verlet", "vel_verlet"]:
        filename = os.path.join(basis_pfad, "output", f"positions_{sim}.npy")
        pos = np.load(filename)
        plot3d_500years(sim, pos)
        chat_gpt_calc_period(planet=9, positions=pos)
        plot_planet_for_years_2d(planet_index=9, positions=pos, simulation=sim)

    # plot3d()
    # calc_period(planet=9)  # Pluto
    # chat_gpt_calc_period(planet=4)
