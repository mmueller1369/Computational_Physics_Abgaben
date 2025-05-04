import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    simulations = ["verlet", "vel_verlet", "euler"]
    for i, sim in enumerate(simulations):
        for sim2 in simulations[i + 1 :]:
            basis_pfad = os.path.dirname(os.path.dirname(__file__))
            filename = os.path.join(basis_pfad, "output", f"positions_{sim}.npy")
            positions = np.load(filename)
            filename2 = os.path.join(basis_pfad, "output", f"positions_{sim2}.npy")
            positions2 = np.load(filename2)

            # planet to compare
            planet = 9

            difference = positions[:, planet, :] - positions2[:, planet, :]
            abs_diff = np.linalg.norm(difference, axis=1)

            plt.figure(figsize=(10, 6))
            plt.plot(abs_diff, label=f"Difference between {sim} and {sim2}")
            plt.xlabel("Time Steps")
            plt.ylabel("Absolute Difference (AU)")
            plt.title(f"Difference in Trajectory for Planet {planet}")
            plt.legend()
            plt.savefig(
                os.path.join(
                    basis_pfad,
                    "output",
                    f"compare_trajectory_planet_{planet}_{sim}_{sim2}.png",
                )
            )
            plt.show()
