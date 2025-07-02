import numpy as np
from numba import njit, prange
import math
import settings


@njit(parallel=True)
def forceH2O(
    x, y, z,
    k_bond, s0, k_angle, theta0, # Intramolecular parameters
    eps, sigma, cutoff, qO, qH, eps0_el, alpha, # Intermolecular parameters
    ):
    gamma_cut = gamma(cutoff, alpha)
    fx = np.zeros(shape=len(x))
    fy = np.zeros(shape=len(x))
    fz = np.zeros(shape=len(x))
    N = len(x)

    e_LJ = 0
    e_coul = 0
    e_bond = 0
    e_angle = 0

    c2 = sigma * sigma / cutoff / cutoff
    c6 = c2 * c2 * c2
    e_LJ_cut = 4.0 * eps * c6 * (c6 - 1.0)

    for mol in prange(N//3):
        # attributing the atom indice
        o = 3*mol  # index of O-atom
        i = 3*mol + 1  # indice of first and second H-atom
        j = 3*mol + 2

        # intramolecular stuff
        # # properties needed
        six = x[o] - x[i]
        siy = y[o] - y[i]
        siz = z[o] - z[i]
        sjx = x[o] - x[j]
        sjy = y[o] - y[j]
        sjz = z[o] - z[j]
        si = math.sqrt(six*six + siy*siy + siz*siz)
        sj = math.sqrt(sjx*sjx + sjy*sjy + sjz*sjz)
        sproj = six*sjx + siy*sjy + siz*sjz
        theta = math.acos(sproj/(si*sj))
        # # updating the forces
        fx[i] += f_bond(i, x, si, k_bond, s0) + f_angle(i, j, x, si, sproj, theta, k_angle, theta0)
        fy[i] += f_bond(i, y, si, k_bond, s0) + f_angle(i, j, y, si, sproj, theta, k_angle, theta0)
        fz[i] += f_bond(i, z, si, k_bond, s0) + f_angle(i, j, z, si, sproj, theta, k_angle, theta0)
        fx[j] += f_bond(j, x, sj, k_bond, s0) + f_angle(j, i, x, sj, sproj, theta, k_angle, theta0)
        fy[j] += f_bond(j, y, sj, k_bond, s0) + f_angle(j, i, y, sj, sproj, theta, k_angle, theta0)
        fz[j] += f_bond(j, z, sj, k_bond, s0) + f_angle(j, i, z, sj, sproj, theta, k_angle, theta0)
        # # updating the energies
        e_bond += k_bond/2 * ((si - s0)**2 + (sj - s0)**2)
        e_angle += k_angle/2 * (theta - theta0)**2

        # intermolecular stuff
        for l in prange(o+3, N):
            # # properties needed
            rolx = x[l] - x[o]
            roly = y[l] - y[o]
            rolz = z[l] - z[o]
            rol = math.sqrt(rolx**rolx + roly**roly + rolz**rolz)
            # # determining absolute values of force (divided by rol) and energy
            if rol > cutoff:
                if l % 3 == 0:  # if l is an O-atom
                    ff_LJol, e_LJol = ffe_LJ(rol, sigma, eps)
                    ql = qO
                else:
                    ff_LJol, e_LJol = 0, 0
                    ql = qH
                ff_coulol, e_coulol = ffe_coul(rol, qO, ql, eps0_el, alpha, cutoff, gamma_cut)
            else:
                ff_LJol, e_LJol = 0, 0
                ff_coulol, e_coulol = 0, 0
            ff_inter = ff_LJol + ff_coulol
            # # updating the forces
            fx[o] -= ff_inter * rolx
            fy[o] -= ff_inter * roly
            fz[o] -= ff_inter * rolz
            fx[l] += ff_inter * rolx
            fy[l] += ff_inter * roly
            fz[l] += ff_inter * rolz
            # # updating the energies
            e_LJ += e_LJol - e_LJ_cut
            e_coul += e_coulol
        
    energies = e_LJ, e_coul, e_bond, e_angle
    return fx, fy, fz, energies


@njit
def f_bond(i, pos, si, k_bond, s0):
    svec = pos[i]
    ff = k_bond * (1 - s0/si)
    return -ff*svec


@njit
def f_angle(i, j, pos, si, sproj, theta, k_angle, theta0):
    sveci = pos[i]
    svecj = pos[j]
    prefac = k_angle * (theta - theta0) / math.tanh(theta)
    vectorial = svecj/sproj - sveci/(2*si**2)
    return -prefac*vectorial


@njit
def ffe_LJ(rol, sigma, eps):
    prefac = 24*eps / rol
    base = sigma/rol
    base6 = base**6
    ff_LJol = prefac * (2*base6**2 - base6) / rol
    e_LJol = 4*eps * (base6**2 - base6)
    return ff_LJol, e_LJol


@njit
def ffe_coul(rol, qo, ql, eps0_el, alpha, cutoff, gamma_cut):
    prefac = qo*ql / (4*math.pi*eps0_el)
    ff_coulol = prefac * (gamma(rol, alpha) - gamma_cut) / rol
    erfcrol = math.erfc(alpha*rol) / rol
    erfccut = math.erfc(alpha*cutoff) / cutoff
    e_coulol = prefac * (erfcrol - erfccut + gamma_cut*(rol*cutoff))
    return ff_coulol, e_coulol


@njit
def gamma(r, alpha):
    summ1 = math.erfc(alpha*r)/r**2
    summ2 = 2*alpha/math.sqrt(math.pi) * math.exp(-alpha**2*r**2)/r
    return summ1 + summ2