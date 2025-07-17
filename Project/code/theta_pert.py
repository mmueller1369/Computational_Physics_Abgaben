import matplotlib.pyplot as plt
import os
import numpy as np
import math
from math import pi
import postprocessing
import settings
# import settings_SI as settings
from scipy.optimize import curve_fit
settings.init()


# plt.plot(step, theta[:,0]*180/np.pi, label="angle")
# plt.show()

# plt.plot(step, si[:,0], label="si")
# plt.plot(step, sj[:,0], label="sj")
# plt.legend()
# plt.show()

# 
k_angle = 75 * 4184 *1e3
reduced_mass = settings.masses[0] * settings.masses[1] / (settings.masses[0] + settings.masses[1])  # in g/mole
omega_exp = math.sqrt(k_angle  *2/ reduced_mass /(0.1e-9)**2 )
print(f"Expected omega: {omega_exp:.2E} Hz")
name = "pert_t_0.1"

energyfile = os.path.join(settings.path, f"part_1/{name}_energy.txt")
energies = np.loadtxt(energyfile).T

# Create subplot for energies: all data and first 100 steps
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

# All data
axs[0].plot(energies[0]*0.5, energies[3], label = r"$E_{bond}$")
axs[0].plot(energies[0]*0.5, energies[4], label = r"$E_{angle}$")
axs[0].plot(energies[0]*0.5, energies[5], label = r"$E_{kin}$")
axs[0].plot(energies[0]*0.5, np.sum(energies[1:], axis=0), label = r"$E_{tot}$")
axs[0].set_title(f'Energies for file {name} (all data)')
axs[0].set_ylabel(r"$E$ [kcal/mole]")
axs[0].legend()

# First 100 steps
t_short = energies[0][:100] * 0.5  # Convert to fs
axs[1].plot(t_short, energies[3][:100], label = r"$E_{bond}$")
axs[1].plot(t_short, energies[4][:100], label = r"$E_{angle}$")
axs[1].plot(t_short, energies[5][:100], label = r"$E_{kin}$")
axs[1].plot(t_short, np.sum(energies[1:, :100], axis=0), label = r"$E_{tot}$")
axs[1].set_title(f'Energies for file {name} (first 100 steps)')
axs[1].set_xlabel(r"$t$ [fs]")
axs[1].set_ylabel(r"$E$ [kcal/mole]")
axs[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(settings.path, f"overleaf_plots/theta/energy_subplot.png"))
plt.show()

# Results bond length perturbation
trajfile = os.path.join(settings.path, f"part_1/{name}_traj.txt")
run = postprocessing.PostprocessingTools(trajfile)
#print([run.data[i].tables['molecules'].properties for i in range(len(run.data))])
steps = range(len(run.data))
si = np.array([run.data[i].tables['molecules']['Si'] for i in steps], dtype=float).flatten()
sj = np.array([run.data[i].tables['molecules']['Sj'] for i in steps], dtype=float).flatten()
theta = np.array([run.data[i].tables['molecules']['Theta'] for i in steps], dtype=float).flatten()
plt.figure(figsize=(10, 5))
plt.plot(steps[:1500], si[:1500], label=r"$s_i$")
plt.axhline(np.mean(si[:1500]), color='r', linestyle='--', label=r"$\langle s_i \rangle$ = " + f"{np.mean(si[:1500]):.5E} nm")
plt.plot(steps[:1500], sj[:1500], label=r"$s_j$")
plt.axhline(np.mean(sj[:1500]), color='g', linestyle='--', label= r"$\langle s_j \rangle$ = " + f"{np.mean(sj[:1500]):.5E} nm")
plt.xlabel(r"$t$ [fs]")
plt.ylabel(r"$s_i$ [nm]")
plt.legend()
plt.title(r"$s_i$ over time with mean value")
plt.savefig(os.path.join(settings.path, f"overleaf_plots/theta/si_sj.png"))
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(steps, theta * 180 / np.pi, label=r"$\Theta$")
plt.axhline(np.mean(theta) * 180 / np.pi, color='r', linestyle='--', label=r"$\langle \Theta \rangle$ = " + f"{np.mean(theta) * 180 / np.pi:.2f}°")
plt.xlabel(r"$t$ [fs]")
plt.ylabel(r"$\Theta$ [°]")
plt.legend()    
plt.title(r"$\Theta$ over time with mean value")
plt.savefig(os.path.join(settings.path, f"overleaf_plots/theta/theta.png"))

# --- Fit beat acoustic function to si for first 200 steps ---
def beat_func(t, A, omega1, omega2, phi, offset):
    # A * cos(omega1 * t + phi) * cos(omega2 * t) + offset
    return A * np.cos(omega1 * t + phi) * np.cos(omega2 * t) + offset

try:
    si_arr = theta[:]#[100:800]  
    t = np.arange(len(si_arr)) * settings.deltat  # Convert steps to time in fs
    # --- Compute average frequency from maxima ---
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(si_arr)
    if len(peaks) > 1:
        peak_times = t[peaks]
        periods = np.diff(peak_times)  # in fs
        avg_period = np.mean(periods)
        avg_freq = 1 / avg_period  # in 1/fs
        print(f"Average period: {avg_period:.2f} fs")
        print(f"Average frequency: {avg_freq:.4f} 1/fs")
        # Optionally, plot the peaks
        plt.figure(figsize=(10, 5))
        plt.plot(t, si_arr, label=r"$s_i$ (data)")
        plt.plot(t[peaks], si_arr[peaks], "rx", label="Maxima")
        plt.xlabel(r"$t$ [fs]")
        plt.ylabel(r"$s_i$ [nm]")
        plt.legend()
        plt.title(r"Detected maxima in $s_i$")
        plt.savefig(os.path.join(settings.path, f"overleaf_plots/theta/si_maxima.png"))
        plt.show()

        # Plot number of maxima vs. position of peak and fit a straight line
        maxima_indices = np.arange(0, len(peak_times))  # 0-based counting
        plt.figure(figsize=(8, 5))
        plt.plot(maxima_indices, peak_times-peak_times[0], 'o', label='Peak positions')
        # Fit straight line
        coeffs, cov = np.polyfit(maxima_indices, peak_times-peak_times[0], 1, cov=True)
        fit_line = np.polyval(coeffs, maxima_indices)
        plt.plot(maxima_indices, fit_line, 'r--', label=f'Fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')
        plt.xlabel('Maxima number')
        plt.ylabel('Peak position [fs]')
        plt.title('Peak position vs. maxima number')
        plt.legend()
        plt.savefig(os.path.join(settings.path, f"overleaf_plots/theta/si.png"))
        plt.show()
        print(f"Fitted line slope (period): {coeffs[0]:.2E} fs per maxima")
        #print(f"Variance of slope: {cov[0,0]:.2E}")
        error = np.sqrt(cov[0,0])
        print(f"Error in slope: {error:.2E} fs per maxima")
        T_plot = coeffs[0]*1e-15
        calculated_omega = 2* pi/ T_plot
        print(f"Calculated omega: {calculated_omega:.2E} Hz")
        err_om = np.sqrt( (2*pi/ T_plot * error)**2) 
        print(f"Error in omega: {err_om:.2E} Hz")


    else:
        print("Not enough maxima found to compute average frequency.")
except Exception as e:
    print("Fit failed:", e)

