import settings
import random
import math
import operator
import debug
import numpy as np


def InitializeAtoms():
    
    nx = 0
    ny = 0
    n = 0
    x = np.zeros(shape=(settings.n1*settings.n2))
    y = np.zeros(shape=(settings.n1*settings.n2))
    vx = np.zeros(shape=(settings.n1*settings.n2))
    vy = np.zeros(shape=(settings.n1*settings.n2))
    while nx < settings.n1:
        ny = 0
        while ny < settings.n2:
            x0 = nx * settings.deltaxyz
            y0 = ny * settings.deltaxyz
            
            vx0 = 0.5 - random.randint(0, 1)
            vy0 = 0.5 - random.randint(0, 1)
                
                
            x[n] = x0
            y[n] = y0
            
            vx[n] = vx0
            vy[n] = vy0
            n += 1
                
            ny += 1
        
        nx += 1
    settings.nparticles = n
        
    # cancel the linear momentum
    svx = np.sum(vx)
    svy = np.sum(vy)
    
    vx -= svx / settings.nparticles
    vy -= svy / settings.nparticles
    svx = np.sum(vx)
    
    # rescale the velocity to the desired temperature
    Trandom = temperature(vx, vy)
    vx, vy = rescalevelocity(vx, vy, settings.Tdesired, Trandom)
    
    # cancel the linear momentum
    svx = np.sum(vx)
    svy = np.sum(vy)
    
    vx -= svx / settings.nparticles
    vy -= svy / settings.nparticles
    svx = np.sum(vx)
    
    return x, y, vx, vy

def temperature(vx, vy):
    # convunits is the conversion factor 
    convunits = 238845.9 # from (gram/mole)*(nm/fs)^2/((kcal/mole)/K) to K
    vsq = 0.
    vsq = np.sum(np.multiply(vx, vx) + np.multiply(vy, vy))
    return settings.mass * vsq / 2./settings.kb / settings.nparticles * convunits
    
    
def rescalevelocity(vx, vy, T1, T2):
    
    vx = vx * math.sqrt(T1 / T2)
    vy = vy * math.sqrt(T1 / T2)
    return vx, vy                      
    
    
    
