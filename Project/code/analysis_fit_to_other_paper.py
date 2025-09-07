import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

N = np.array([9, 20, 37, 51, 94, 237, 471, 960])
r = np.array([3.59, 5.71, 6.30, 7.14, 8.69, 11.84, 14.86, 18.88])/10

def fit_func(N, a):
    return a * N**(1/3)

popt, pcov = curve_fit(fit_func, N, r)
a_fit = popt[0]
N_fit = np.linspace(N.min(), N.max(), 100)

plt.plot(N, r, 'o', label='data from article', color='black')
plt.plot(N_fit, fit_func(N_fit, a_fit), '-', label=rf'fit: $r = {a_fit:.2f}\cdot N^{{1/3}}$', color='orange')
plt.axvline(216, ls='--', label=r'$N=216$', color='blue')
plt.xlabel(r'$N$')
plt.ylabel(r'$r$ [nm]')
plt.legend()
plt.savefig('radius_article.png', dpi=300, bbox_inches='tight')
plt.show()

print(f'r_216: {fit_func(216, a_fit)} +- {np.sqrt(pcov[0, 0]) * 216**(1/3)} nm')


mH = 1.6735e-27
mO = 2.6561e-26
mMol = 2 * mH + mO

mbulk = 996.52 # kg/m^3
rhoMol = mbulk/mMol # molekules/m^3
rhoMol /= 1e27 # molekules/nm^3
rhoO = rhoMol # particles/nm^3
rhoH = 2*rhoMol # particles/nm^3

print(f'rho_O: {rhoO:.2f} particles/nm^3')
print(f'rho_H: {rhoH:.2f} particles/nm^3')