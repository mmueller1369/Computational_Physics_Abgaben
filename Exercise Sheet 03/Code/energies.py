import numpy as np
import os
import matplotlib.pyplot as plt


def read_energy(filename):
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()
    # Skip header
    data = []
    for line in lines[1:]:
        if line.strip() == "":
            continue
        parts = line.split()
        step = int(parts[0])
        values = [float(x) for x in parts[1:]]
        data.append([step] + values)
    return np.array(data)


def plot_energies(data):
    plt.plot(data[:, 0], data[:, 1], label="Potential Energy")
    plt.plot(data[:, 0], data[:, 2], label="Kinetic Energy")
    total_energy = data[:, 1] + data[:, 2]
    plt.plot(data[:, 0], total_energy, label="Total Energy")
    plt.xlabel("Timeste")
    plt.ylabel("Energy in Joule")
    plt.title("Energy Plot MD Simulation")
    plt.legend()
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "output", "energy.png"
    )
    plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    # Read the energy data from the file
    data = read_energy("energy_prod.txt")
    # Plot the energies
    plot_energies(data)
