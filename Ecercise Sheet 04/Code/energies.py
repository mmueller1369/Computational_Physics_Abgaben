import numpy as np
import os
import matplotlib.pyplot as plt

multiplier = 1 / 4.1868e-06  # kcal*fs/gram/nm to nm/fs
multiplier = 475081.0162102793
multiplier = 4.1868e5
# multiplier = 2.39e5


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
    plt.plot(data[:, 0], data[:, 1], label="Potential Energy kcal/mol")
    plt.plot(data[:, 0], data[:, 2], label="Kinetic Energy kcal/mol")
    total_energy = data[:, 1] + data[:, 2]
    print("pot mean:", data[:, 1].mean())
    print("kin mean:", data[:, 2].mean())
    plt.plot(data[:, 0], total_energy, label="Total Energy")
    plt.xlabel("Timeste")
    plt.ylabel("Energy kcal/mol")
    plt.title("Energy Plot MD Simulation")
    plt.legend()
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "output", "energy.png"
    )
    plt.savefig(path)
    plt.show()


# def plot_energies(data):
#     multipliers = np.logspace(5, 6, 100)  # exponents 5 to 6 (i.e., 1e5 to 1e6)
#     var = np.array(
#         [np.var(data[:, 1] + data[:, 2] * multiplier) for multiplier in multipliers]
#     )
#     min_index = np.argmin(var)
#     best_multiplier = multipliers[min_index]
#     print("Multiplier zum Minimum von var:", best_multiplier)
#     plt.semilogx(multipliers, var)
#     plt.axvline(best_multiplier)

#     plt.show()


if __name__ == "__main__":
    # Read the energy data from the file
    data = read_energy("energy_prod.txt")
    # Plot the energies
    plot_energies(data)
