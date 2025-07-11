import forces
import matplotlib.pyplot as plt
import numpy as np

T = 300
kB = 1.38064852e-23
e = 1.602176621e-19
eps0_el = 8.854187817e-12
sigma = 3.21e-10
eps = 0.2*kB*T
qO = -0.84*e
qH = 0.42*e
cutoff = 2.5*sigma
alpha = 1/cutoff
gamma_cut = forces.gamma(cutoff, alpha)
J_to_kcpm = 1/4184*6.0022140857e23

import settings_SI as settings
settings.init()
kB = settings.kb
eps0_el = settings.eps0_el
sigma = settings.sigma
eps = settings.eps
qO = settings.qO
qH = settings.qH
cutoff = settings.cutoff
alpha = settings.alpha
gamma_cut = forces.gamma(cutoff, alpha)


rols = np.linspace(0.5, 2.5)*sigma

ffsC = np.zeros((len(rols), 3))
esC = np.zeros((len(rols), 3))
ffsLJ = np.zeros((len(rols), 3))
esLJ = np.zeros((len(rols), 3))
for i, (q1, q2) in enumerate([[qO, qO],
                              [qO, qH],
                              [qH, qH]]):
    for j, rol in enumerate(rols):
        ffsC[j,i], esC[j,i] = forces.ffe_coul(rol, q1, q2,
                                            eps0_el, alpha,
                                            cutoff, gamma_cut)
        if i == 0:
            ffsLJ[j,i], esLJ[j,i] = forces.ffe_LJ(rol, sigma, eps)
        else:
            ffsLJ[j,i], esLJ[j,i] = 0, 0
names = ["OO", "OH", "HH"]

plt.figure(figsize=(10, 6))
for i in range(3):
    # plt.plot(rols/sigma, ffsC[:,i]*rols, label=rf"$F_{{Coul}} {names[i]}$")
    # plt.plot(rols/sigma, ffsLJ[:,i]*rols, label=rf"$F_{{LJ}} {names[i]}$")
    plt.plot(rols/sigma, (ffsC[:,i]+ffsLJ[:,i])*rols, label=rf"$F_{{tot}} {names[i]}$")
plt.xlabel(r"$r/\sigma$")
plt.ylabel(r"$F$ [(kcal/mole)/nm]")
plt.legend()
plt.savefig('kcpm.png')
# plt.ylim(-1, 10)
plt.show()

# plt.figure(figsize=(10, 6))
# for i in range(3):
    # plt.plot(rols/sigma, esC[:,i]*rols, label=rf"$E_{{Coul}} {names[i]}$")
    # plt.plot(rols/sigma, esLJ[:,i]*rols, label=rf"$E_{{LJ}} {names[i]}$")
    # plt.plot(rols/sigma, (esC[:,i]+esLJ[:,i])*rols, label=rf"$E_{{tot}} {names[i]}$")
# plt.xlabel(r"$r/\sigma$")
# plt.ylabel(r"$E$ [kcal/mole]")
# plt.legend()
# plt.show()