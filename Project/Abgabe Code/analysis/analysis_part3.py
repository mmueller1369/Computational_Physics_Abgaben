import matplotlib.pyplot as plt
import os
import numpy as np
from postprocessing import PostprocessingTools
import settings
settings.init()


folder = "part_3"
every_nth_frame_post = 10
smooth_hist = 3
cutoff = settings.cutoff


# ------------- Energies moving into equilibrium ------------- #
energyfileeq = os.path.join(settings.path, f"{folder}/energy_eq.txt")
energyfile = os.path.join(settings.path, f"{folder}/energy.txt")
energieseq = np.loadtxt(energyfileeq).T
energies = np.loadtxt(energyfile).T

timeeq = (energieseq[0] - energieseq[0,-1]) / 1e3
epoteq = np.sum(energieseq[1:5], axis=0)
epotintraeq = np.sum(energieseq[3:5], axis=0)
epotintereq = np.sum(energieseq[1:3], axis=0)
ekineq = energieseq[5]
etoteq = np.sum(energieseq[1:], axis=0)

time = energies[0] / 1e3
epot = np.sum(energies[1:5], axis=0)
epotintra = np.sum(energies[3:5], axis=0)
epotinter = np.sum(energies[1:3], axis=0)
ekin = energies[5]
etot = np.sum(energies[1:], axis=0)

plt.plot(timeeq, epotintraeq, label=r'$E_{pot, intra}$', color='xkcd:orange')
plt.plot(time, epotintra, color='xkcd:orange')
plt.plot(timeeq, epotintereq, label=r'$E_{pot, inter}$', color='red')
plt.plot(time, epotinter, color='red')
# plt.plot(timeeq, epoteq, label=r'$E_{pot, tot}$')
# plt.plot(time, epot)
plt.plot(timeeq, ekineq, label=r'$E_{kin}$', color='blue')
plt.plot(time, ekin, color='blue')
plt.plot(timeeq, etoteq, label=r'$E_{tot}$', color='black')
plt.plot(time, etot, color='black')
plt.axvline(timeeq[-1], ls='dashed', color='black')
plt.xlabel(r"$t$ [ps]")
plt.ylabel(r"$E$ [kcal/mole]")
plt.legend()
plt.savefig(os.path.join(settings.path, f"images/{folder}_energies.png"), dpi=300, bbox_inches='tight')
plt.show()



# ------------- Temperature moving into equilibrium ------------- #
tempfileeq = os.path.join(settings.path, f"{folder}/temp_eq.txt")
tempfile = os.path.join(settings.path, f"{folder}/temp.txt")
tempfileeq = np.loadtxt(tempfileeq).T
tempfile = np.loadtxt(tempfile).T

plt.plot(timeeq, tempfileeq[1], label = r"$T$", color='blue')
plt.plot(time, tempfile[1], color='blue')
plt.axvline(timeeq[-1], ls='dashed', color='black')
plt.xlabel(r"$t$ [ps]")
plt.ylabel(r"$T$ [K]")
# plt.legend()
plt.savefig(os.path.join(settings.path, f"images/{folder}_temperature.png"), dpi=300, bbox_inches='tight')
plt.show()



# ------------- Droplet properties moving into equilibrium ------------- #
trajfileeq = os.path.join(settings.path, f"{folder}/traj_eq.txt")
trajfile = os.path.join(settings.path, f"{folder}/traj.txt")
eq = PostprocessingTools(trajfileeq, every_nth_frame=every_nth_frame_post, cutoff=cutoff)
prod = PostprocessingTools(trajfile, every_nth_frame=every_nth_frame_post, cutoff=cutoff)
stepeq, rgseq, bseq, paramseq, pcovseq = eq.calculate_droplet_properties(smooth_hist=smooth_hist)
step, rgs, bs, params, pcovs = prod.calculate_droplet_properties(smooth_hist=smooth_hist)
step += stepeq[-1]
tteq = timeeq[::every_nth_frame_post]
tt = time[::every_nth_frame_post]
rseq = paramseq[:,0]
bulkseq = paramseq[:,1]
sharpnesseq = paramseq[:,2]
urseq = np.sqrt(pcovseq[:,0,0])
ubulkseq = np.sqrt(pcovseq[:,1,1])
usharpnesseq = np.sqrt(pcovseq[:,2,2])
rs = params[:,0]
bulks = params[:,1]
sharpness = params[:,2]
urs = np.sqrt(pcovs[:,0,0])
ubulks = np.sqrt(pcovs[:,1,1])
usharpness = np.sqrt(pcovs[:,2,2])

plt.plot(tteq, rseq, label=r"$R$", color='blue')
plt.plot(tt, rs, color='blue')
plt.fill_between(tteq, rseq+urseq, rseq-urseq, color='blue', alpha=.2)
plt.fill_between(tt, rs+urs, rs-urs, color='blue', alpha=.2)
plt.plot(tteq, rgseq, label=r"$R_g$", color='red')
plt.plot(tt, rgs, color='red')
plt.axvline(tteq[-1], ls='dashed', color='black')
plt.xlabel(r"$t$ [ps]")
plt.ylabel(r"$R$ [nm]")
plt.legend()
plt.savefig(os.path.join(settings.path, f"images/{folder}_radii.png"), dpi=300, bbox_inches='tight')
plt.show()

plt.plot(tteq, bulkseq, label=r"$\hat\rho$", color='purple')
plt.plot(tt, bulks, color='purple')
plt.fill_between(tteq, bulkseq+ubulkseq, bulkseq-ubulkseq, color='purple', alpha=.2)
plt.fill_between(tt, bulks+ubulks, bulks-ubulks, color='purple', alpha=.2)
plt.axvline(tteq[-1], ls='dashed', color='black')
plt.xlabel(r"$t$ [ps]")
plt.ylabel(r"$\hat\rho$ [particles/nm$^3$]")
# plt.legend()
plt.savefig(os.path.join(settings.path, f"images/{folder}_bulkdensity.png"), dpi=300, bbox_inches='tight')
plt.show()

plt.plot(tteq, sharpnesseq, label=r"$a$", color='orange')
plt.plot(tt, sharpness, color='orange')
plt.fill_between(tteq, sharpnesseq+usharpnesseq, sharpnesseq-usharpnesseq, color='orange', alpha=.2)
plt.fill_between(tt, sharpness+usharpness, sharpness-usharpness, color='orange', alpha=.2)
plt.axvline(tteq[-1], ls='dashed', color='black')
plt.xlabel(r"$t$ [ps]")
plt.ylabel(r"sharpness [nm]")
# plt.legend()
plt.savefig(os.path.join(settings.path, f"images/{folder}_sharpness.png"), dpi=300, bbox_inches='tight')
plt.show()

plt.plot(tteq, bseq, label = r"$b$", color='green')
plt.plot(tt, bs, color='green')
plt.axvline(tteq[-1], ls='dashed', color='black', label=r'$t_{eq}$')
plt.xlabel(r"$t$ [ps]")
plt.ylabel(r"$b$")
# plt.legend()
plt.savefig(os.path.join(settings.path, f"images/{folder}_asphericity.png"), dpi=300, bbox_inches='tight')
plt.show()


# ------------- ACF ------------- #
def autocorr(x):
    x = np.asarray(x)
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]
    return result / result[0]

dt = tt[1] - tt[0]
labels = [r"$R$", r"$\hat\rho$", r"$a$", r"$R_g$", r"$b$"]
series = [rs, bulks, sharpness, rgs, bs]
colors = ['blue', 'purple', 'orange', 'red', 'green']
taus = []

for s, label, color in zip(series, labels, colors):
    acf = autocorr(s)
    t = np.arange(len(acf)) * dt
    plt.plot(t, acf, label=label, color=color)
    tau_idx = np.where(acf < 1/np.e)[0][0]
    tau = t[tau_idx]
    taus.append(tau)
    plt.axvline(tau, color=color, ls='dashed', alpha=0.5)
    print(f"Autocorrelation time for {label}: {taus[-1]:.2f} ps")
plt.axhline(1/np.e, color='black', ls='dashed', label=r'$1/e$')
plt.xlabel(r'$t$ [ps]')
plt.ylabel('autocorrelation')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(settings.path, f"images/{folder}_ACF.png"), dpi=300, bbox_inches='tight')
plt.show()