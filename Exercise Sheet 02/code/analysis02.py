from data_io import import_data
from tools import compute_ke, compute_pe
import matplotlib.pyplot as plt
import numpy as np

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
