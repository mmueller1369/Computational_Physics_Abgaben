import settings
import numpy as np
import math
import sys


def WriteEnergy(fileenergy, itime, epot, ekin, vx2, vy2, vz2):

    fileenergy.write("%i %e %e %e %e %e\n" % (itime, epot, ekin, vx2, vy2, vz2))


def WriteTrajectory(fileoutput, itime, x, y, z, vx, vy, vz, fx, fy, fz):

    fileoutput.write("ITEM: TIMESTEP \n")
    fileoutput.write("%i \n" % itime)
    fileoutput.write("ITEM: NUMBER OF ATOMS \n")
    fileoutput.write("%i \n" % (settings.n1 * settings.n2 * settings.n3))
    fileoutput.write("ITEM: BOX BOUNDS \n")
    fileoutput.write("%e %e \n" % (settings.xlo, settings.xhi))
    fileoutput.write("%e %e \n" % (settings.ylo, settings.yhi))
    fileoutput.write("%e %e \n" % (settings.zlo, settings.zhi))
    fileoutput.write("ITEM: ATOMS id type x y z vx vy vz fx fy fz\n")

    for i in range(0, len(x)):
        fileoutput.write(
            "%i %i %e %e %e %e %e %e %e %e %e\n"
            % (
                i,
                i,
                (x[i] % (settings.xhi - settings.xlo)),
                (y[i] % (settings.yhi - settings.ylo)),
                (z[i] % (settings.zhi - settings.zlo)),
                vx[i],
                vy[i],
                vz[i],
                fx[i],
                fy[i],
                fz[i],
            )
        )


def inputset():
    return (
        settings.xlo,
        settings.xhi,
        settings.ylo,
        settings.yhi,
        settings.zlo,
        settings.zhi,
        settings.eps,
        settings.sigma,
        settings.cutoff,
        settings.deltat,
        settings.mass,
    )


def squarevelocity(vx, vy, vz, mass):
    vx2 = 0
    vy2 = 0
    vz2 = 0
    i = 0
    for i in range(0, len(vx)):
        vx2 += vx[i] ** 2
        vy2 += vy[i] ** 2
        vz2 += vz[i] ** 2
    return 0.5 * mass * vx2, 0.5 * mass * vy2, 0.5 * mass * vz2
