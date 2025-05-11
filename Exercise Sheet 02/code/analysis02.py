from data_io import import_data
from tools import compute_ke, compute_pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import params
import os


# exercise e
for dt in [1, 10, 0.1]:
    filename = f"exercise_e_dt={dt}.dat"
    data, properties, box_bounds = import_data(filename)
    multiplier = (
        10  # otherwise the computation takes very long, so we skip some data points
    )
    time = np.arange(
        0, data.shape[0] * dt * params.dt_export, dt * multiplier * params.dt_export
    )
    np.save(os.path.join(params.path, f"exercise_e_time_dt={dt}.npy"), time)
    ke = np.array(
        [compute_ke(data, t * multiplier) for t in range(data.shape[0] // multiplier)]
    )
    np.save(os.path.join(params.path, f"exercise_e_ke_dt={dt}.npy"), ke)
    pe = np.array(
        [
            compute_pe(
                data,
                t * multiplier,
                potential="lj_cut",
                potential_params=[0.297741315, 0.188, 2.5],
                boundary_conditions=params.boundary_conditions,
                box_bounds=params.box_bounds,
            )
            for t in range(data.shape[0] // multiplier)
        ]
    )
    np.save(os.path.join(params.path, f"exercise_e_pe_dt={dt}.npy"), pe)
    e = ke + pe
    np.save(os.path.join(params.path, f"exercise_e_e_dt={dt}.npy"), e)

    plt.plot(time, ke, label="Kinetic Energy")
    plt.plot(time, pe, label="Potential Energy")
    plt.plot(time, e, label="Total Energy")
    plt.xlabel(r"Time [$\tau$]")
    plt.ylabel(r"Energy [$\epsilon$]")
    plt.legend()
    plt.semilogy()
    plt.savefig(os.path.join(params.path, f"exercise_e_e_dt={dt}.png"))
    plt.close()

    ke_mean = np.mean(ke)
    pe_mean = np.mean(pe)
    e_mean = np.mean(e)
    ke_var = np.var(ke)
    pe_var = np.var(pe)
    e_var = np.var(e)
    ke_std = np.std(ke)
    pe_std = np.std(pe)
    e_std = np.std(e)

    print(f"dt={dt}:")
    print(f"  KE - Mean: {ke_mean}, Std Dev: {ke_std}, Variance: {ke_var}")
    print(f"  PE - Mean: {pe_mean}, Std Dev: {pe_std}, Variance: {pe_var}")
    print(f"  E  - Mean: {e_mean}, Std Dev: {e_std}, Variance: {e_var}")

    vx = data[:: (multiplier * params.dt_export), properties["vx"], :]
    vy = data[:: (multiplier * params.dt_export), properties["vy"], :]
    vx_sq_sum = np.sum(vx**2, axis=1)
    vy_sq_sum = np.sum(vy**2, axis=1)
    np.save(os.path.join(params.path, f"exercise_e_vx_sq_sum_dt={dt}.npy"), vx_sq_sum)
    np.save(os.path.join(params.path, f"exercise_e_vy_sq_sum_dt={dt}.npy"), vy_sq_sum)

    plt.plot(time, vx_sq_sum, label=r"$\sum_i{v_{x,i}^2}$")
    plt.plot(time, vy_sq_sum, label=r"$\sum_i{v_{y,i}^2}$")
    plt.xlabel(r"Time [$\tau$]")
    plt.ylabel(r"$\sum_i{v_{\alpha,i}^2}$ [$\sigma/\tau^2$]$")
    plt.legend()
    plt.savefig(os.path.join(params.path, f"exercise_e_v_dt={dt}.png"))
    plt.close()


# exercise f
N = np.array([10, 20, 30, 40, 50]) ** 2
t_N = np.array([1 / 19.07, 1.22, 4.0, 12.31, 30.15]) * 100  # s/iteration


# make the fit
def power_law(N, alpha, c):
    return c * N**alpha


popt, pcov = curve_fit(power_law, N, t_N)
alpha, c = popt
print(f"Fitted alpha: {alpha}, fitted c: {c}")
print(
    f"Time for 1000 iterations for n=1000: {power_law(1000**2, alpha, c)*1000/3600/24/365} years"
)
# plot the fit
N_fit = np.linspace(min(N), max(N), 100)
t_N_fit = power_law(N_fit, alpha, c)
plt.figure()
plt.plot(N, t_N, "o", label="Data")
plt.plot(N_fit, t_N_fit, "-", label=f"Fit: $t_N \propto N^{{{alpha:.2f}}}$")
plt.xlabel(r"$N$")
plt.ylabel(r"$t_N$")
plt.legend()
plt.loglog()
plt.savefig(os.path.join(params.path, "exercise_f_fit.png"))
plt.close()


# exercise g
filename_g = "exercise_f_nparticles=20.dat"
data_g, properties_g, box_bounds_g = import_data(filename_g)
# Compute v^2 = vx^2 + vy^2 for all particles and time steps and flatten the array; neglect the first 10 time steps as the thermostat didn't act yet
vx = data_g[10:, properties_g["vx"], :]
vy = data_g[10:, properties_g["vy"], :]
v_squared = vx**2 + vy**2
v_squared_flat = v_squared.flatten()
# Create a histogram to estimate the distribution P(v^2)
hist, bin_edges = np.histogram(v_squared_flat, bins=20, density=True)
np.save(os.path.join(params.path, "exercise_g_hist.npy"), hist)
np.save(os.path.join(params.path, "exercise_g_bin_edges.npy"), bin_edges)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
plt.figure()
plt.bar(
    bin_centers,
    hist,
    width=bin_edges[1] - bin_edges[0],
    align="center",
    label=r"$P(v^2)$",
)

plt.xlabel(r"$\sum_i{v_{i}^2}$ [$\sigma/\tau^2$]")
plt.ylabel(r"$P(v^2)$")
plt.legend()
plt.savefig(os.path.join(params.path, "exercise_g_hist.png"))
plt.close()


# exercise h
for cutoff in [3.25, 4.0]:
    filename = f"exercise_h_cut={cutoff}.dat"
    data, properties, box_bounds = import_data(filename)
    multiplier = 10  # otherwise the computation takes very long
    time = np.arange(0, data.shape[0] * params.dt_export, params.dt_export * multiplier)
    np.save(os.path.join(params.path, f"exercise_h_time_cut={cutoff}.npy"), time)
    ke = np.array(
        [compute_ke(data, t * multiplier) for t in range(data.shape[0] // multiplier)]
    )
    np.save(os.path.join(params.path, f"exercise_h_ke_cut={cutoff}.npy"), ke)
    pe = np.array(
        [
            compute_pe(
                data,
                t * multiplier,
                potential="lj_cut",
                potential_params=[0.297741315, 0.188, 2.5],
                boundary_conditions=params.boundary_conditions,
                box_bounds=params.box_bounds,
            )
            for t in range(data.shape[0] // multiplier)
        ]
    )
    np.save(os.path.join(params.path, f"exercise_h_pe_cut={cutoff}.npy"), pe)
    e = ke + pe
    np.save(os.path.join(params.path, f"exercise_h_e_cut={cutoff}.npy"), e)

    plt.plot(time, ke, label="Kinetic Energy")
    plt.plot(time, pe, label="Potential Energy")
    plt.plot(time, e, label="Total Energy")
    plt.xlabel(r"Time [$\tau$]")
    plt.ylabel(r"Energy [$\epsilon$]")
    plt.legend()
    plt.semilogy()
    plt.savefig(os.path.join(params.path, f"exercise_h_e_cut={cutoff}.png"))
    plt.close()
