from data_io import import_data
from tools import compute_ke, compute_pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# exercise e
for dt in [1, 10, 0.1]:
    filename = f"exercise_e_dt={dt}.dat"
    data, properties, box_bounds = import_data(filename)
    time = data.shape[0] * dt
    ke = np.array([compute_ke(data, t) for t in range(data.shape[0])])
    pe = np.array(
        [
            compute_pe(
                data, t, potential="lj_cut", potential_params=[0.297741315, 0.188, 2.5]
            )
            for t in range(data.shape[0])
        ]
    )
    e = ke + pe

    plt.plot(time, ke, label="Kinetic Energy")
    plt.plot(time, pe, label="Potential Energy")
    plt.plot(time, e, label="Total Energy")
    plt.xlabel("Time [fs]")
    plt.ylabel("Energy [kcal/mol]; might be wrong, recheck")
    plt.legend()

    ke_mean = np.mean(ke)
    pe_mean = np.mean(pe)
    e_mean = np.mean(e)
    ke_var = np.var(ke)
    pe_var = np.var(pe)
    e_var = np.var(e)

    vx = data[:, properties["vx"], :]
    vy = data[:, properties["vy"], :]
    vx_sq_sum = np.sum(vx**2, axis=1)
    vy_sq_sum = np.sum(vy**2, axis=1)

    plt.plot(time, vx_sq_sum, label=r"$\sum_i{v_{x,i}^2}$")
    plt.plot(time, vy_sq_sum, label=r"$\sum_i{v_{y,i}^2}$")
    plt.xlabel("Time [fs]")
    plt.ylabel(
        r"$\sum_i{v_{\alpha,i}^2}$ [$\text{nm}^2$/$\text{fs}^2$]$; might be wrong, recheck"
    )
    plt.legend()


# exercise f
N = np.array([10, 20, 30, 40, 50]) ** 2
t_N = np.array([1, 2, 4, 8, 16])  # to be changed


# make the fit
def power_law(N, alpha, c):
    return c * N**alpha


popt, pcov = curve_fit(power_law, N, t_N)
alpha, c = popt
print(f"Fitted alpha: {alpha}")
# plot the fit
N_fit = np.linspace(min(N), max(N), 100)
t_N_fit = power_law(N_fit, alpha, c)
plt.figure()
plt.plot(N, t_N, "o", label="Data")
plt.plot(N_fit, t_N_fit, "-", label=f"Fit: $t_N \propto N^{{{alpha:.2f}}}$")
plt.xlabel("N")
plt.ylabel("t_N")
plt.legend()
plt.show()


# exercise g
filename_g = "exercise_f_nparticles=20.dat"
data_g, properties_g, box_bounds_g = import_data(filename_g)
# Compute v^2 = vx^2 + vy^2 for all particles and time steps and flatten the array; neglect the first 10 time steps as the thermostat didn't act yet
vx = data_g[10:, properties_g["vx"], :]
vy = data_g[10:, properties_g["vy"], :]
v_squared = vx**2 + vy**2
v_squared_flat = v_squared.flatten()
# Create a histogram to estimate the distribution P(v^2)
hist, bin_edges = np.histogram(v_squared_flat, bins=50, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
plt.figure()
plt.plot(bin_centers, hist, label=r"$P(v^2)$")
plt.xlabel(r"$v^2$ [$\text{nm}^2/\text{fs}^2$]")
plt.ylabel(r"$P(v^2)$")
plt.legend()
plt.show()


# exercise h
for cutoff in [3.25, 4.0]:
    filename = f"exercise_h_cutoff={cutoff}.dat"
    data, properties, box_bounds = import_data(filename)
    time = data.shape[0] * 1
    ke = np.array([compute_ke(data, t) for t in range(data.shape[0])])
    pe = np.array(
        [
            compute_pe(
                data,
                t,
                potential="lj_cut",
                potential_params=[0.297741315, 0.188, cutoff],
            )
            for t in range(data.shape[0])
        ]
    )
    e = ke + pe

    plt.plot(time, ke, label="Kinetic Energy")
    plt.plot(time, pe, label="Potential Energy")
    plt.plot(time, e, label="Total Energy")
    plt.xlabel("Time [fs]")
    plt.ylabel("Energy [kcal/mol]; might be wrong, recheck")
    plt.legend()
