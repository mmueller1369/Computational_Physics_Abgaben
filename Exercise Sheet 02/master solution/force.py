import settings
import numpy as np
from numba import njit, prange


@njit(parallel=True)
#@njit

#### unit of the force: (kcal/mole)/nm

def forceLJ(x, y, xlo, xhi, ylo, yhi, eps, r0, cutoff):
    
    fx = np.zeros(shape=len(x))
    fy = np.zeros(shape=len(x))
    N = len(x)
    
    i = 0
    sf2a = r0*r0 / cutoff / cutoff
    sf6a = sf2a * sf2a * sf2a
    epotcut = 8.*settings.eps*sf6a*(sf6a - 1.)
    epot = 0
#    for i in range(N-1):
    for i in prange(N-1):
        j = i + 1
#        for j in range(i+1,N):
        for j in prange(i+1, N):
            rijx = pbc(x[i], x[j],xlo, xhi)
            rijy = pbc(y[i], y[j],ylo, yhi)
            
            r2 = rijx * rijx + rijy * rijy 
            # calculate fx, fy, fz
            if r2 < cutoff * cutoff:
                sf2 = r0*r0 / r2
                sf6 = sf2 * sf2 * sf2
                epot += (8.*eps*sf6*(sf6 - 1.)) #-epotcut)
                ff = 48.*eps*sf6*(sf6 - 0.5)/r2
                fx[i] -= ff*rijx
                fy[i] -= ff*rijy
                
                fx[j] += ff*rijx
                fy[j] += ff*rijy
                
    return fx, fy, epot
  
@njit  
def pbc(xi, xj, xlo, xhi):
    
    l = xhi-xlo
    
    xi = xi % l
    xj = xj % l
    
    rij = xj - xi  
    if abs(rij) > 0.5*l:
        rij = rij - np.sign(rij) * l 
        
    return rij
