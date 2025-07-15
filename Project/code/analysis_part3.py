import matplotlib.pyplot as plt
import os
import numpy as np
from postprocessing import PostprocessingTools
import settings
# import settings_SI as settings
settings.init()


# folder = "part_3"


def analyze(folder, smooth_hist=5):
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
    plt.axvline(timeeq[-1], ls='dashed', color='black', label=r'$t_{eq}$')
    plt.xlabel(r"$t$ [ps]")
    plt.ylabel(r"$E$ [kcal/mole]")
    plt.legend()
    plt.show()



    # ------------- Temperature moving into equilibrium ------------- #
    tempfileeq = os.path.join(settings.path, f"{folder}/temp_eq.txt")
    tempfile = os.path.join(settings.path, f"{folder}/temp.txt")
    tempfileeq = np.loadtxt(tempfileeq).T
    tempfile = np.loadtxt(tempfile).T

    plt.plot(timeeq, tempfileeq[1], label = r"$T$", color='blue')
    plt.plot(time, tempfile[1], color='blue')
    plt.axvline(timeeq[-1], ls='dashed', color='black', label=r'$t_{eq}$')
    plt.xlabel(r"$t$ [ps]")
    plt.ylabel(r"$T$ [K]")
    plt.legend()
    plt.show()



    # ------------- Droplet radius moving into equilibrium ------------- #
    trajfileeq = os.path.join(settings.path, f"{folder}/traj_eq.txt")
    trajfile = os.path.join(settings.path, f"{folder}/traj.txt")
    every_nth_frame_post = 1
    eq = PostprocessingTools(trajfileeq, every_nth_frame=every_nth_frame_post)
    prod = PostprocessingTools(trajfile, every_nth_frame=every_nth_frame_post)
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
    plt.axvline(tteq[-1], ls='dashed', color='black', label=r'$t_{eq}$')
    plt.xlabel(r"$t$ [ps]")
    plt.ylabel(r"$R$ [nm]")
    plt.legend()
    plt.show()

    plt.plot(tteq, bulkseq, label=r"$R$", color='blue')
    plt.plot(tt, bulks, color='blue')
    plt.fill_between(tteq, bulkseq+ubulkseq, bulkseq-ubulkseq, color='blue', alpha=.2)
    plt.fill_between(tt, bulks+ubulks, bulks-ubulks, color='blue', alpha=.2)
    plt.axvline(tteq[-1], ls='dashed', color='black', label=r'$t_{eq}$')
    plt.xlabel(r"$t$ [ps]")
    plt.ylabel(r"$\rho_\text{bulk}$ [particles/nm$^3$]")
    plt.legend()
    plt.show()

    plt.plot(tteq, sharpnesseq, label=r"$R$", color='blue')
    plt.plot(tt, sharpness, color='blue')
    plt.fill_between(tteq, sharpnesseq+usharpnesseq, sharpnesseq-usharpnesseq, color='blue', alpha=.2)
    plt.fill_between(tt, sharpness+usharpness, sharpness-usharpness, color='blue', alpha=.2)
    plt.axvline(tteq[-1], ls='dashed', color='black', label=r'$t_{eq}$')
    plt.xlabel(r"$t$ [ps]")
    plt.ylabel(r"sharpness [nm]")
    plt.legend()
    plt.show()

    plt.plot(tteq, bseq, label = r"$b$", color='blue')
    plt.plot(tt, bs, color='blue')
    plt.axvline(tteq[-1], ls='dashed', color='black', label=r'$t_{eq}$')
    plt.xlabel(r"$t$ [ps]")
    plt.ylabel(r"$b$")
    plt.legend()
    plt.show()


    # --- Autokorrelationsfunktionen und -zeiten --- #
    def autocorr(x):
        x = np.asarray(x)
        x = x - np.mean(x)
        result = np.correlate(x, x, mode='full')
        result = result[result.size // 2:]
        return result / result[0]

    dt = tt[1] - tt[0]
    labels = [r"$R_g$", r"$R$", r"$b$", r"$\rho_\mathrm{bulk}$", r"sharpness"]
    series = [rgs, rs, bs, bulks, sharpness]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    taus = []
    plt.figure(figsize=(8,5))
    for s, label, color in zip(series, labels, colors):
        acf = autocorr(s)
        t = np.arange(len(acf)) * dt
        plt.plot(t, acf, label=label, color=color)
        # 1/e-Korrelationszeit bestimmen
        tau_idx = np.where(acf < 1/np.e)[0][0]
        tau = t[tau_idx]
        taus.append(tau)
        plt.axvline(tau, color=color, ls='dashed', alpha=0.5)
        print(f"Autocorrelation time for {label}: {taus[-1]:.2f} ps")
    plt.xlabel(r'Lag [ps]')
    plt.ylabel('Autocorrelation')
    plt.title('Autokorrelationsfunktionen der Strukturgrößen')
    plt.legend()
    plt.tight_layout()
    plt.show()


analyze("part_3")