import numpy as np
from numba import njit, prange
import math


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
    fx_inter = np.zeros(shape=len(x))
    fy_inter = np.zeros(shape=len(x))
    fz_inter = np.zeros(shape=len(x))
    fx_intra = np.zeros(shape=len(x))
    fy_intra = np.zeros(shape=len(x))
    fz_intra = np.zeros(shape=len(x))
    N = len(x)

    e_LJ = 0.0
    e_coul = 0.0
    e_bond = 0.0
    e_angle = 0.0

    c2 = sigma * sigma / cutoff / cutoff
    c6 = c2 * c2 * c2
    e_LJ_cut = 4.0 * eps * c6 * (c6 - 1.0)

    for mol in range(N//3):
        # attributing indices
        o = 3*mol
        i = 3*mol + 1
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
        # # calculating the bond forces which update the H and O-atom forces
        f_bondix = f_bond(six, si, k_bond, s0)
        f_bondiy = f_bond(siy, si, k_bond, s0)
        f_bondiz = f_bond(siz, si, k_bond, s0)
        f_bondjx = f_bond(sjx, sj, k_bond, s0)
        f_bondjy = f_bond(sjy, sj, k_bond, s0)
        f_bondjz = f_bond(sjz, sj, k_bond, s0)
        f_angleix = f_angle(six, sjx, si, sproj, theta, k_angle, theta0)
        f_angleiy = f_angle(siy, sjy, si, sproj, theta, k_angle, theta0)
        f_angleiz = f_angle(siz, sjz, si, sproj, theta, k_angle, theta0)
        f_anglejx = f_angle(sjx, six, sj, sproj, theta, k_angle, theta0)
        f_anglejy = f_angle(sjy, siy, sj, sproj, theta, k_angle, theta0)
        f_anglejz = f_angle(sjz, siz, sj, sproj, theta, k_angle, theta0)
        # # updating the forces
        fx_intra[o] = -(f_bondix + f_bondjx + f_angleix + f_anglejx)
        fy_intra[o] = -(f_bondiy + f_bondjy + f_angleiy + f_anglejy)
        fz_intra[o] = -(f_bondiz + f_bondjz + f_angleiz + f_anglejz)
        fx_intra[i] = f_bondix + f_angleix
        fy_intra[i] = f_bondiy + f_angleiy
        fz_intra[i] = f_bondiz + f_angleiz
        fx_intra[j] = f_bondjx + f_anglejx
        fy_intra[j] = f_bondjy + f_anglejy
        fz_intra[j] = f_bondjz + f_anglejz
        # # updating the energies
        e_bond += k_bond/2 * ((si - s0)**2 + (sj - s0)**2)
        e_angle += k_angle/2 * (theta - theta0)**2

        # intermolecular stuff
        for atom1, q1 in zip([o, i, j], [qO, qH, qH]):
            for atom2 in range(atom1+1, N):
                # only consider interactions between atoms of different molecules
                if atom2//3 != mol:
                    # # properties needed
                    rolx = x[atom2] - x[atom1]
                    roly = y[atom2] - y[atom1]
                    rolz = z[atom2] - z[atom1]
                    rol = math.sqrt(rolx*rolx + roly*roly + rolz*rolz)
                    # # only for distances smaller than cutoff
                    if rol < cutoff:
                        # # LJ only for O-O
                        if atom1%3 == 0 and atom2%3 == 0:
                            ff_LJol, e_LJol = ffe_LJ(rol, sigma, eps)
                        else:
                            ff_LJol, e_LJol = 0, 0
                        # # Coulomb interaction for all
                        q2 = qO if atom2 % 3 == 0 else qH
                        ff_coulol, e_coulol = ffe_coul(rol, q1, q2, eps0_el,
                                                       alpha, cutoff, gamma_cut)
                    else:
                        ff_LJol, e_LJol = 0, 0
                        ff_coulol, e_coulol = 0, 0
                    ff_inter = ff_LJol + ff_coulol
                    # update forces
                    fx_inter[atom1] -= ff_inter * rolx
                    fy_inter[atom1] -= ff_inter * roly
                    fz_inter[atom1] -= ff_inter * rolz
                    fx_inter[atom2] += ff_inter * rolx
                    fy_inter[atom2] += ff_inter * roly
                    fz_inter[atom2] += ff_inter * rolz
                    # update energies
                    e_LJ += e_LJol - e_LJ_cut if ff_LJol != 0 else 0
                    e_coul += e_coulol
    if abs(np.sum(fx_inter)) > 1e-10:
        print("x inter:", np.sum(fx_inter))
    for i in prange(N):
        fx[i] = fx_intra[i] + fx_inter[i]
        fy[i] = fy_intra[i] + fy_inter[i]
        fz[i] = fz_intra[i] + fz_inter[i]
    energies = e_LJ, e_coul, e_bond, e_angle
    return fx, fy, fz, energies


@njit
def f_bond(svec, si, k_bond, s0):
    ff = k_bond * (1 - s0/si)
    return ff*svec


@njit
def f_angle(sveci, svecj, si, sproj, theta, k_angle, theta0):
    prefac = - k_angle * (theta - theta0) / math.tan(theta)
    vectorial = svecj/sproj - sveci/si**2
    return prefac*vectorial


@njit
def ffe_LJ(rol, sigma, eps):
    prefac = 24*eps / rol
    base = sigma / rol
    base2 = base*base
    base6 = base2*base2*base2
    ff_LJol = prefac * base6 * (2*base6 - 1) / rol
    e_LJol = 4*eps * base6 * (base6 - 1)
    return ff_LJol, e_LJol
    # return 0, 0


@njit
def ffe_coul(rol, qo, ql, eps0_el, alpha, cutoff, gamma_cut):
    prefac = qo*ql / (4*math.pi*eps0_el)
    ff_coulol = prefac * (gamma(rol, alpha) - gamma_cut) / rol
    erfcrol = math.erfc(alpha*rol) / rol
    erfccut = math.erfc(alpha*cutoff) / cutoff
    e_coulol = prefac * (erfcrol - erfccut + gamma_cut*(rol - cutoff))
    return ff_coulol, e_coulol
    # return 0, 0


@njit
def gamma(r, alpha):
    summ1 = math.erfc(alpha*r)/r**2
    summ2 = 2*alpha/math.sqrt(math.pi) * math.exp(-alpha**2*r**2)/r
    return summ1 + summ2