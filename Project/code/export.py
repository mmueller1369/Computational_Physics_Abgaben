import settings
import tools


def WriteEnergy(file, time, vx, vy, vz, e_LJ, e_coul, e_bond, e_angle):
    ekin = tools.computeKineticEnergy(vx, vy, vz, settings.masses)
    file.write("%i %e %e %e %e %e\n" % (time, e_LJ, e_coul, e_bond, e_angle, ekin))


def WriteTemperature(file, itime, vx, vy, vz):
    temp = tools.computeTemperature(vx, vy, vz, settings.masses)
    file.write("%i %e\n" % (itime, temp))


def WriteGr(file, itime, hist):
    file.write("%i" % itime)
    for value in hist:
        file.write("%e " % value)
    file.write("\n")


def WriteTrajectory(file, itime, x, y, z, vx, vy, vz, fx, fy, fz):
    natoms = len(x)

    file.write("ITEM: TIMESTEP \n")
    file.write("%i \n" % itime)

    file.write("ITEM: NUMBER OF ATOMS \n")
    file.write("%i \n" % natoms)

    file.write("ITEM: BOX BOUNDS \n")
    file.write("%e %e \n" % (settings.bounds[0,0], settings.bounds[0,1]))
    file.write("%e %e \n" % (settings.bounds[1,0], settings.bounds[1,1]))
    file.write("%e %e \n" % (settings.bounds[2,0], settings.bounds[2,1]))

    file.write("ITEM: ATOMS id type x y z vx vy vz fx fy fz\n")
    for i in range(natoms):
        itype = "O" if i%3 == 0 else "H"
        file.write(
            "%i %s %e %e %e %e %e %e %e %e %e\n"
            % (i, itype, x[i], y[i], z[i], vx[i], vy[i], vz[i], fx[i], fy[i], fz[i])
        )