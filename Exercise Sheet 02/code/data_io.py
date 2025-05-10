import params
import numpy as np
import os


def export_data(
    data: np.ndarray,
    selected_properties: list = True,
    dt_export: int = params.dt_export,
    filename: str = params.filename,
    box_bounds: tuple = params.box_bounds,
    path: str = params.path,
    mode: str = "normal",
) -> None:
    """
    Export data to a file in LAMMPS dump format.

    Parameters:
        data (numpy.ndarray): The data to export, shape (timesteps, properties, particles).
        selected_properties (list): The list of properties to include in the output. If True, all properties from params.properties are included.
    Originally taken from params:
        dt_export (int): Only export data every dt_export steps.
        filename (str): The name of the output file.
        box_bounds (tuple): The bounds of the simulation box.
        path (str): The directory where the file will be saved. If None, the current directory is used.
        mode (str): The mode of the export. Options: "normal", "debug". In "debug" mode, the function will overwrite existing files without asking for confirmation.

    Returns:
        None
    """
    if path:
        filename = os.path.join(path, filename)

    if os.path.exists(filename) and not mode == "debug":
        user_input = input(f"The file '{filename}' already exists. Overwrite? (y/n): ")
        if user_input != "y":
            print("Export cancelled.")
            return

    with open(filename, "w") as f:
        num_timesteps = data.shape[0]
        num_particles = data.shape[2]

        if selected_properties:
            selected_properties = list(params.properties.keys())
        else:
            for prop in selected_properties:
                if prop not in params.properties:
                    raise ValueError(
                        f"Property '{prop}' not found in property mapping."
                    )

        for t in range(0, num_timesteps, dt_export):
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{t}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{num_particles}\n")
            f.write("ITEM: BOX BOUNDS\n")
            for bound in box_bounds:
                f.write(f"{bound[0]} {bound[1]}\n")
            f.write("ITEM: ATOMS " + " ".join(selected_properties) + "\n")
            for p in range(num_particles):
                values = [
                    data[t, params.properties[name], p] for name in selected_properties
                ]
                f.write(" ".join(map(str, values)) + "\n")
    print(f"Data successfully written to '{filename}'.")


def import_data(filename: str, path: str = params.path) -> tuple:
    """
    Import data from a file in LAMMPS dump format.

    Parameters:
        filename (str): The name of the input file.
    Originally taken from params:
        path (str): The directory where the file is located. If None, the current directory is used.

    Returns:
        tuple: A tuple containing:
            - data (numpy.ndarray): The imported data, shape (timesteps, properties, particles).
            - properties (dict): A dictionary mapping property names to their indices.
            - box_bounds (list): The bounds of the simulation box for each timestep.
    """
    if path:
        filename = os.path.join(path, filename)

    with open(filename, "r") as f:
        lines = f.readlines()

    timesteps = []
    box_bounds = []
    data = []
    properties = {}

    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            timestep = int(lines[i + 1].strip())
            timesteps.append(timestep)
            i += 2
        elif lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            num_particles = int(lines[i + 1].strip())
            i += 2
        elif lines[i].startswith("ITEM: BOX BOUNDS"):
            bounds = []
            for _ in range(3):  # Assuming 3 dimensions
                bounds.append(tuple(map(float, lines[i + 1].strip().split())))
                i += 1
            box_bounds.append(bounds)
            i += 1
        elif lines[i].startswith("ITEM: ATOMS"):
            if not properties:
                property_names = lines[i].strip().split()[2:]
                properties = {name: idx for idx, name in enumerate(property_names)}
            timestep_data = []
            for _ in range(num_particles):
                values = list(map(float, lines[i + 1].strip().split()))
                timestep_data.append(values)
                i += 1
            data.append(timestep_data)
        else:
            i += 1

    data = np.array(data)
    data = np.transpose(
        data, (0, 2, 1)
    )  # Reshape to (timesteps, properties, particles)

    return data, properties, box_bounds
