import forces
import numpy as np
import matplotlib.pyplot as plt
import settings
# import settings_SI as settings
settings.init()

# settings.eps0_el = 1

gamma_cut = forces.gamma(settings.cutoff, settings.alpha)
rols = np.linspace(settings.cutoff*.05, settings.cutoff, 100)
ffsC = np.zeros((len(rols), 3))
esC = np.zeros((len(rols), 3))
ffsLJ = np.zeros((len(rols), 3))
esLJ = np.zeros((len(rols), 3))
for i, (q1, q2) in enumerate([[settings.qO, settings.qO],
                              [settings.qH, settings.qH],
                              [settings.qO, settings.qH]]):
    for j, rol in enumerate(rols):
        ffsC[j,i], esC[j,i] = forces.ffe_coul(rol, q1, q2,
                                            settings.eps0_el, settings.alpha,
                                            settings.cutoff, gamma_cut)
        if i == 0:
            ffsLJ[j,i], esLJ[j,i] = forces.ffe_LJ(rol, settings.sigma, settings.eps)
        else:
            ffsLJ[j,i], esLJ[j,i] = 0, 0
names = ["OO", "OH", "HH"]

plt.figure(figsize=(10, 6))
for i in range(3):
    # plt.plot(rols/settings.sigma, ffsC[:,i]*rols, label=rf"$F_{{Coul}} {names[i]}$")
    # plt.plot(rols/settings.sigma, ffsLJ[:,i]*rols, label=rf"$F_{{LJ}} {names[i]}$")
    plt.plot(rols/settings.sigma, (ffsC[:,i]+ffsLJ[:,i])*rols, label=rf"$F_{{tot}} {names[i]}$")
plt.xlabel(r"$r/\sigma$")
plt.ylabel(r"$F$ [(kcal/mole)/nm]")
plt.legend()
# plt.ylim(-3, 10)
plt.show()

# plt.figure(figsize=(10, 6))
# for i in range(3):
#     plt.plot(rols/settings.sigma, esC[:,i]*rols, label=rf"$E_{{Coul}} {names[i]}$")
#     plt.plot(rols/settings.sigma, esLJ[:,i]*rols, label=rf"$E_{{LJ}} {names[i]}$")
#     plt.plot(rols/settings.sigma, (esC[:,i]+esLJ[:,i])*rols, label=rf"$E_{{tot}} {names[i]}$")
# plt.xlabel(r"$r/\sigma$")
# plt.ylabel(r"$E$ [kcal/mole]")
# plt.legend()
# plt.show()
