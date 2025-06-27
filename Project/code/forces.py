import settings
import numpy as np
from numba import njit, prange
import math


# unit of the force: (kcal/mole)/nm

# no pbcs used!
@njit(parallel=True)
def forceH2O(x, y, z, paramsLJ, paramsCoul, paramsInter):
    eps, sigma = paramsLJ
    qO, qH, alpha, cutoff = paramsCoul
    gamma_cut = gamma(cutoff, alpha)
    k_bond, k_angle, s0, theta0 = paramsInter
    fx = np.zeros(shape=len(x))
    fy = np.zeros(shape=len(x))
    fz = np.zeros(shape=len(x))
    N = len(x)

    e_LJ = 0
    e_coul = 0
    e_bond = 0
    e_angle = 0

    for mol in prange(N//3):
        # attributing the atom indice
        o = 3*mol  # index of O-atom
        i = 3*mol + 1  # indice of first and second H-atom
        j = 3*mol + 2

        # intramolecular stuff
        # # properties needed
        six = x[i] - x[o]
        siy = y[i] - y[o]
        siz = z[i] - z[o]
        sjx = x[j] - x[o]
        sjy = y[j] - y[o]
        sjz = z[j] - z[o]
        si = math.sqrt(six**six + siy**siy + siz**siz)
        sj = math.sqrt(sjx**sjx + sjy**sjy + sjz**sjz)
        sproj = six**sjx + siy**sjy + siz**sjz
        theta = math.arccos(sproj/(si*sj))
        # # updating the forces
        fx[i] -= f_bond(i, x, si, k_bond, s0) + f_angle(i, j, x, si, sproj, theta, k_angle, theta0)
        fy[i] -= f_bond(i, y, si, k_bond, s0) + f_angle(i, j, y, si, sproj, theta, k_angle, theta0)
        fz[i] -= f_bond(i, z, si, k_bond, s0) + f_angle(i, j, z, si, sproj, theta, k_angle, theta0)
        fx[j] -= f_bond(j, x, sj, k_bond, s0) + f_angle(j, i, x, sj, sproj, theta, k_angle, theta0)
        fy[j] -= f_bond(j, y, sj, k_bond, s0) + f_angle(j, i, y, sj, sproj, theta, k_angle, theta0)
        fz[j] -= f_bond(j, z, sj, k_bond, s0) + f_angle(j, i, z, sj, sproj, theta, k_angle, theta0)
        # # updating the energies
        e_bond += k_bond/2 * ((si - s0)**2 - (sj - s0)**2)
        e_angle += k_angle/2 * (theta - theta0)**2

        # intermolecular stuff
        for l in prange(o+3, N):
            # # properties needed
            rolx = x[l] - x[o]
            roly = y[l] - y[o]
            rolz = z[l] - z[o]
            rol = math.sqrt(rolx**rolx + roly**roly + rolz**rolz)
            if l % 3 == 0:  # if l is an O-atom
                ff_LJol, e_LJol = ffe_LJ(rol, sigma, eps)
                ql = qO
            else:
                ff_LJol, e_LJol = 0, 0
                ql = qH
            ff_coulol, e_coulol = ffe_coul(
                qO, ql, rol, alpha, cutoff, gamma_cut)
            ff_inter = ff_LJol + ff_coulol
            # # updating the forces
            fx[o] -= ff_inter * rolx
            fy[o] -= ff_inter * roly
            fz[o] -= ff_inter * rolz
            fx[l] += ff_inter * rolx
            fy[l] += ff_inter * roly
            fz[l] += ff_inter * rolz
            # # updating the energies
            e_LJ += e_LJol
            e_coul += e_coulol

    return fx, fy, fz, e_LJ, e_coul, e_bond, e_angle


@njit
def f_bond(i, pos, si, k_bond, s0):
    svec = pos[i]
    return k_bond * (si - s0) * svec/si


@njit
def f_angle(i, j, pos, si, sproj, theta, k_angle, theta0):
    sveci = pos[i]
    svecj = pos[j]
    prefac = k_angle * (theta - theta0) / math.tanh(theta)
    vectorial = svecj/sproj - sveci/(2*si**2)
    return prefac*vectorial


@njit
def ffe_LJ(rol, sigma, eps):
    prefac = 24*eps / rol
    base = sigma/rol
    base6 = base**6
    ff_LJol = prefac * (2*base6**2 - base6) / rol
    e_LJol = 4*eps * (base6**2 - base6)
    return ff_LJol, e_LJol


@njit
def gamma(r, alpha):
    summ1 = math.erfc(alpha*r)/r**2
    summ2 = 2*alpha/math.sqrt(math.pi) * math.exp(-alpha**2*r**2)/r
    return summ1 + summ2


@njit
def ffe_coul(qo, ql, rol, alpha, cutoff, gamma_cut):
    prefac = qo*ql / (4*math.pi*settings.eps0)
    ff_coulol = prefac * (gamma(rol, alpha) - gamma_cut) / rol
    erfcrol = math.erfc(alpha*rol) / rol
    erfccut = math.erfc(alpha*cutoff) / cutoff
    e_coulol = prefac * (erfcrol - erfccut + gamma_cut*(rol*cutoff))
    return ff_coulol, e_coulol
