import numpy as np
import params
from forces import gravitation_potential
from scipy.signal import find_peaks

def read_properties_from_ovito(ovito_filepath: str, pIDs: list, columns: list) -> [np.ndarray, np.ndarray]:
    """
    Reads properties from an ovito file
    Returns the time (in the units of the system) and the corresponding property for each timestep
    -----------
    Parameters:
    ovito_filepath: path of the ovito file
    columns: list of columns to be read
    pIDs: list of wanted particle IDs
    """
    file = open(ovito_filepath, 'r')
    lines = file.readlines()
    nvalues = params.tmax // params.tsave
    time = np.array([params.tsave * i for i in range(nvalues)])
    properties = np.zeros(nvalues, len(pIDs), len(columns))
    for t in range(nvalues):
        for pID in pIDs:
            line_split = lines[8 + t * (9 + params.nparticles) + pID].split()
            properties[t] = [float(line_split[j]) for j in columns]
    return time, properties



def compute_distance_from_ovito(ovito_filepath: str, pID_1: int, pID_2: int) -> [np.ndarray, np.ndarray]:
    """
    Computes the distance between two given particles at a given timestep
    Returns the time (in the units of the system) and the corresponding computed distance for each timestep from the file
    -----------
    Parameters:
    ovito_filepath: path of the ovito file
    pID_1, 2: corresponding partices IDs
    """
    time, properties = read_properties_from_ovito(ovito_filepath, [pID_1, pID_2], [2,3,4])
    distance = np.zeros(len(properties))
    for i in range(len(properties)):
        x1, y1, z1 = properties[i,0,::]
        x2, y2, z2 = properties[i,1,::]
        distance[i] = np.linalg.norm(np.array([x2 - x1, y2 - y1, z2 - z1]))
    return time, distance


# def compute_distance_from_ovito(ovito_filepath: str, pID_1: int, pID_2: int) -> [np.ndarray, np.ndarray]:
#     """
#     Computes the distance between two given particles at a given timestep
#     Returns the time (in the units of the system) and the corresponding computed distance for each timestep from the file
#     -----------
#     Parameters:
#     ovito_filepath: path of the ovito file
#     pID_1, 2: corresponding partices IDs
#     """
#     file = open(ovito_filepath, 'r')
#     lines = file.readlines()
#     nvalues = params.tmax // params.tsave
#     time = np.array([params.tsave * params.dt * i for i in range(nvalues)])
#     distance = np.zeros(nvalues)
#     for timestep in range(nvalues):
#         line_1 = lines[8 + timestep * (9 + params.nparticles) + pID_1]
#         line_2 = lines[8 + timestep * (9 + params.nparticles) + pID_2]
#         x1, y1, z1 = [float(value) for value in line_1.split()[2:5]]
#         x2, y2, z2 = [float(value) for value in line_2.split()[2:5]]
#         distance[timestep] = np.linalg.norm(np.array([x2 - x1, y2 - y1, z2 - z1]))
#     return time, distance


def sinus_model(t, a, omega, phi):
    # help function for orbit properties
    return a * np.sin(omega * t + phi)

def compute_orbit_properties_from_ovito(ovito_filepath: str, pID: int) -> [float, float, float]:
    """
    Computes the mean distance, period length and eccentricity of the orbit of a given particle around the sun
    Returns the computed properties
    -----------
    Parameters:
    ovito_filepath: path of the ovito file
    pID: corresponding partice ID
    """
    time, distance = compute_distance_from_ovito(ovito_filepath, 1, pID)
    
    peaks, _ = find_peaks(distance)
    peak_times = time[peaks]
    periods = np.diff(peak_times)
    period_length = np.mean(periods)
    
    dist_mean = np.mean(distance)
    
    dist_min = np.min(distance)
    dist_max = np.max(distance)
    exc = (dist_max - dist_min) / (dist_max + dist_min)
    
    return dist_mean, period_length, exc


def compute_com(ovito_filepath: str, masses: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Computes the center of mass of the system for each timestep
    Returns the time (in the units of the system) and the corresponding com vector for each timestep from the file
    -----------
    Parameters:
    ovito_filepath: path of the ovito file
    masses: array containing the particle masses
    """
    file = open(ovito_filepath, 'r')
    lines = file.readlines()
    nvalues = params.tmax // params.tsave
    time = np.array([params.tsave * params.dt * i for i in range(nvalues)])
    com = np.zeros(nvalues, 3)
    total_mass = np.sum(masses)
    for timestep in range(nvalues):
        for i in range(params.nparticles):
            line = lines[8 + timestep * (9 + params.nparticles) + i + 1]
            position = np.array([float(value) for value in line.split()[2:5]])
            com[timestep] += masses[i] * position
        com[timestep] = com[timestep] / total_mass
    return time, com
    


def compute_pe(ovito_filepath: str, masses: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Computes the potential energy of the system at each timestep from the file
    Returns the time (in the units of the system) and the corresponding potential energy at each timestep
    -----------
    Parameters:
    ovito_filepath: path of the ovito file
    masses: array containing the particle masses
    """
    time, _ = compute_distance_from_ovito(ovito_filepath, 1, 2)
    pe = np.zeros(len(time))
    G = 1.488136e-5
    # iteration uses the fact that it is a pair potential and thus symmetrical
    for i in range(params.nparticles):
        for j in range(i):
            _, distance = compute_distance_from_ovito(ovito_filepath, i, j)
            pe += 2 * (- G * masses[i] * masses[j] / distance)
    return time, pe



def compute_ke(ovito_filepath: str, masses: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Computes the kinetic energy of the system at each timestep from the file
    Returns the time (in the units of the system) and the corresponding kinetic energy at each timestep
    -----------
    Parameters:
    ovito_filepath: path of the ovito file
    masses: array containing the particle masses
    """
    pIDs = np.array([i + 1 for i in range(params.nparticles)])
    time, velocities = read_properties_from_ovito(ovito_filepath, pIDs, [5,6,7])
    ke = np.zeros(len(time))
    for t in range(len(time)):
        velocities_t = np.array(np.linalg.norm(vel) for vel in velocities[t,::,::])
        ke[t] = np.sum(1 / 2 * masses * velocities_t * velocities_t)
    return time, ke



def compute_e(ovito_filepath: str, masses: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Computes the total energy of the system at each timestep from the file
    Returns the time (in the units of the system) and the corresponding energy at each timestep
    -----------
    Parameters:
    ovito_filepath: path of the ovito file
    masses: array containing the particle masses
    """
    time, ke = compute_ke(ovito_filepath, masses)
    _, pe = compute_pe(ovito_filepath, masses)
    e = ke + pe
    return time, e

compute_orbit_properties_from_ovito('output_euler.dat', 10)
