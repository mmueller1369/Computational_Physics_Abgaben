import matplotlib.pyplot as plt
import os
import numpy as np
from postprocessing import PostprocessingTools, fermi_distribution
import settings
settings.init()


folder = "part_4"

trajfileeq = os.path.join(settings.path, f"{folder}/traj_eq.txt")
trajfile = os.path.join(settings.path, f"{folder}/traj.txt")
prod = PostprocessingTools(trajfile)

r, rho, params, pcov = prod.calculate_rho()
rH, rhoH, paramsH, pcovH = prod.calculate_rho(only_hatoms=True)
rhoH /= 2  # divide by 2 because we only have H atoms in the histogram
_, rgs, asphericities, _, _ = prod.calculate_droplet_properties()
mean_rg = np.mean(rgs)
uncertainty_rg = np.std(rgs)
mean_b = np.mean(asphericities)
uncertainty_b = np.std(asphericities)
radius = params[0]
uradius = np.sqrt(pcov[0][0])
bulk = params[1]
ubulk = np.sqrt(pcov[1][1])
sharp = params[2]
usharp = np.sqrt(pcov[2][2])
radiusH = paramsH[0]
uradiusH = np.sqrt(pcovH[0][0])
bulkH = paramsH[1]
ubulkH = np.sqrt(pcovH[1][1])
sharpH = paramsH[2]
usharpH = np.sqrt(pcovH[2][2])


# -------------- Parameters -------------- #
print(f"Radius = {radius} +- {uradius}\n \
Bulk density = {bulk} +- {ubulk}\n \
Sharpness = {sharp} +- {usharp}\n \
Radius H = {radiusH} +- {uradiusH}\n \
Bulk density H = {bulkH} +- {ubulkH}\n \
Sharpness H = {sharpH} +- {usharpH}\n \
Radius of Gyration = {mean_rg} +- {uncertainty_rg/np.sqrt(2000)}\n \
Asphericity = {mean_b} +- {uncertainty_b/np.sqrt(2000)}")


# -------------- Density -------------- #
rfit = np.linspace(0,settings.rmax_hist, 100)
plt.plot(r, rho, color='red', label=r'$\langle\rho_O(r)\rangle_t$')
plt.plot(rH, rhoH, color='gray', label=r'$\langle\rho(r)_H\rangle_t$ / 2')
plt.axhline(bulk, label=r'$\hat\rho_O$', color='purple')
# plt.fill_between(rfit,
#                  bulk+ubulk, bulk-ubulk, color='purple', alpha=.2)
plt.axvline(radius, label=r'$R_O$', color='blue')
# plt.fill_between([radius+uradius, radius-uradius],
#                  0, 40, color='blue', alpha=.2)
plt.axvline(radius+sharp, label=r'$R_O \pm a_O$', color='orange')
plt.axvline(radius-sharp, color='orange')
plt.plot(rfit, fermi_distribution(rfit, *params), label=r'$\langle\rho_O(r)\rangle_t$ fit', color='black')
plt.legend()
plt.xlabel(r'$r$ [nm]')
plt.ylabel(r'$\langle\rho\rangle_t$ [particles/nm$^3$]')
# plt.xlim(0.1,settings.rmax_hist)
# plt.ylim(0,38)
plt.savefig(os.path.join(settings.path, f"images/{folder}_density.png"), dpi=300, bbox_inches='tight')
plt.show()


# -------------- Dipole Projection -------------- #
dist, value = prod.calculate_dipole_projections_by_distance()
fig, ax2 = plt.subplots()
color2 = 'black'
ax2.set_xlabel(r'$r$ [nm]')
ax2.set_ylabel(r'$\langle\rho\rangle_t$ [particles/nm$^3$]')
ln2 = ax2.plot(rfit, fermi_distribution(rfit, *params), label=r'$\langle\rho_O(t)\rangle_t$ fit', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

ax1 = ax2.twinx()
color1 = 'green'
ax1.set_ylabel(r'$\langle p_\parallel\rangle_t$', color=color1)
ln1 = ax1.plot(dist, value, label=r'$\langle p_\parallel\rangle_t$', color=color1)
ax1.axhline(0, label=r'$p_\parallel = 0$', color=color1, ls='--')
ln3 = plt.plot([0], [0], label=r'$p_\parallel = 0$', color=color1, ls='--')
ax1.tick_params(axis='y', labelcolor=color1)

lns = ln1 + ln2 + ln3
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='center left')
plt.savefig(os.path.join(settings.path, f"images/{folder}_dipoleprojection.png"), dpi=300, bbox_inches='tight')
plt.show()
all_projections = prod.calculate_dipole_projections()
all_projections_flat = np.concatenate(all_projections)
print(f"Mean projection: {np.mean(all_projections_flat)} +- {np.std(all_projections_flat)/np.sqrt(len(all_projections_flat))}")        

# -------------- Electrostatic Field and Energy -------------- #
dist, pot, field = prod.calculate_electrostatic_potential_and_field(dr_hist=settings.dr_hist)
distfit, potfit, fieldfit = prod.calculate_electrostatic_potential_and_field(
    fitted_densities=(rfit,
                      fermi_distribution(rfit, *paramsH),
                      fermi_distribution(rfit, *params)))
fig, ax1 = plt.subplots()
color1 = 'blue'
ax1.set_xlabel(r'$r$ [nm]')
ax1.axhline(0, color=color1, ls='dashed')
ax1.set_ylabel(r'$U_\text{el}$ [kcal/mol]', color=color1)
ln1 = ax1.plot(dist, pot, color=color1, label=r'$U_\text{el}$')
# ln2 = ax1.plot(rfit, potfit, color=color1, label=r'$U_\text{el} fit$', ls='--')
ax1.axvline(radius, color='black', ls='dashed', label=r'$R$')
ln2 = ax1.plot([0], [0], label=r'$R$', color='black', ls='dashed')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'red'
ax2.set_ylabel(r'$E_\text{el}$ [kcal/mol/nm]', color=color2)
ax2.axhline(0, color=color2, ls='dashed')
ln3 = ax2.plot(dist, field, color=color2, label=r'$E_\text{el}$')
# ln4 = ax1.plot(rfit, fieldfit, color=color2, label=r'$E_\text{el} fit$', ls='--')
ax2.tick_params(axis='y', labelcolor=color2)

lns = ln1 + ln3 + ln2
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='lower left')
plt.savefig(os.path.join(settings.path, f"images/{folder}_electricstuff.png"), dpi=300, bbox_inches='tight')
plt.show()


# -------------- Droplet Freezing -------------- #
steps, temperature = prod.calculate_droplet_temperature()
droplet_size = [prod.data[s].particles.count//3 for s in steps]
tempfileeq = os.path.join(settings.path, f"{folder}/temp.txt")
time, temperature_all = np.loadtxt(tempfileeq).T
time /= 1e3

fig, ax1 = plt.subplots()
color1 = 'black'
ax1.set_xlabel(r'$t$ [ps]')
ax1.set_ylabel(r'$T$ [K]', color=color1)
ln1 = ax1.plot(time, temperature, label=r"$T_\text{droplet}$", color='green')
ln2 = ax1.plot(time, temperature_all, label=r"$T_\text{all}$", color='blue', lw=.5)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'orange'
ax2.set_ylabel(r'$N_\text{droplet}$', color=color2)
ln3 = ax2.plot(time, droplet_size, color='orange', label=r'$N_\text{droplet}$')
ax2.tick_params(axis='y', labelcolor=color2)

lns = ln1 + ln2 + ln3
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='best')
plt.savefig(os.path.join(settings.path, f"images/{folder}_freeze.png"), dpi=300, bbox_inches='tight')
plt.show()


fig, ax1 = plt.subplots()
color1 = 'black'
ax1.set_xlabel(r'$t$ [ps]')
ax1.set_ylabel(r'$T_\text{droplet}-T_\text{all}$ [K]', color=color1)
ln1 = ax1.plot(time, temperature-temperature_all, label=r"$T_\text{droplet}-T_\text{all}$", color='black')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'orange'
ax2.set_ylabel(r'$N_\text{droplet}$', color=color2)
ln2 = ax2.plot(time, droplet_size, color='orange', label=r'$N_\text{droplet}$')
ax2.tick_params(axis='y', labelcolor=color2)

lns = ln1 + ln2
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='best')
plt.savefig(os.path.join(settings.path, f"images/{folder}_freezediff.png"), dpi=300, bbox_inches='tight')
plt.show()