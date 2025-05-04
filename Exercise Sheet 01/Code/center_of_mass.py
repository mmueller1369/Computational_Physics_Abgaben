import numpy as np
import os
import matplotlib.pyplot as plt
import my_io

BASIS_PFAD = os.path.dirname(os.path.dirname(__file__))


def center_of_mass(simulation="euler"):
    filename = os.path.join(BASIS_PFAD, "output", f"positions_{simulation}.npy")
    positions = np.load(filename)
    mass = my_io.read_masses(
        "mass.dat"
    )  # in units of 10^29 kg sun is first element and pluto last
    mass = mass.reshape(-1, 1)  # reshape mass to be a column vector

    # Compute the center of mass for each time step
    total_mass = np.sum(mass)  # Total mass (scalar)
    com = np.sum(positions * mass, axis=1) / total_mass  # Shape: (n_steps, 3)
    return com


def plot3d(Rcom, simulation="euler"):
    # 3D plot for the center of mass trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x = Rcom[:, 0]  # Extract x-coordinates
    y = Rcom[:, 1]  # Extract y-coordinates
    z = Rcom[:, 2]  # Extract z-coordinates

    ax.plot(x, y, z, label="Center of Mass Trajectory", color="blue")
    ax.scatter(x, y, z, c=np.arange(len(x)), cmap="viridis", s=30)  # Optional scatter
    ax.set_xlabel("X Position (AU)")
    ax.set_ylabel("Y Position (AU)")
    ax.set_zlabel("Z Position (AU)")
    ax.set_title(f"3D Trajectory of Center of Mass ({simulation})")
    ax.legend()
    plt.savefig(
        os.path.join(BASIS_PFAD, "output", f"center_of_mass_traj_{simulation}.png")
    )
    plt.show()


def plot_subplots(Rcom, simulation="euler"):
    # Subplots for top, front, and side views
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    x = Rcom[:, 0]  # Extract x-coordinates
    y = Rcom[:, 1]  # Extract y-coordinates
    z = Rcom[:, 2]  # Extract z-coordinates

    # Top view (X-Y plane)
    axs[0].plot(x, y, color="blue")
    axs[0].set_title("Top View (X-Y Plane)")
    axs[0].set_xlabel("X Position (AU)")
    axs[0].set_ylabel("Y Position (AU)")

    # Front view (X-Z plane)
    axs[1].plot(x, z, color="green")
    axs[1].set_title("Front View (X-Z Plane)")
    axs[1].set_xlabel("X Position (AU)")
    axs[1].set_ylabel("Z Position (AU)")

    # Side view (Y-Z plane)
    axs[2].plot(y, z, color="red")
    axs[2].set_title("Side View (Y-Z Plane)")
    axs[2].set_xlabel("Y Position (AU)")
    axs[2].set_ylabel("Z Position (AU)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(BASIS_PFAD, "output", f"center_of_mass_subplots_{simulation}.png")
    )
    plt.show()


if __name__ == "__main__":
    for x in ["euler", "verlet", "vel_verlet"]:
        print("start")
        R = center_of_mass(x)
        print("R berechnet")
        plot3d(R, x)  # Plot the center of mass trajectory
        plot_subplots(R, x)  # Plot subplots for top, front, and side views
        print("plots erstellt")
