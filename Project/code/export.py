import settings
# import settings_SI as settings
import tools
settings.init()


def WriteEnergy(file, step, vx, vy, vz, e_LJ, e_coul, e_bond, e_angle, masses):
    ekin = tools.computeKineticEnergy(vx, vy, vz, masses)
    file.write("%i %e %e %e %e %e\n" % (step*settings.deltat,
        e_LJ, e_coul, e_bond, e_angle, ekin))
    # file.write("%i %e %e %e %e %e\n" % (time,
    #     e_LJ*settings.conv_energy, e_coul*settings.conv_energy, e_bond*settings.conv_energy, e_angle*settings.conv_energy, ekin*settings.conv_energy))


def WriteTemperature(file, step, vx, vy, vz, masses):
    temp = tools.computeTemperature(vx, vy, vz, masses)
    file.write("%i %e\n" % (step*settings.deltat, temp))


def WriteTrajectory(file, step, x, y, z, vx, vy, vz, fx, fy, fz, masses):
    natoms = len(x)

    file.write("ITEM: TIMESTEP \n")
    file.write("%i \n" % step)

    file.write("ITEM: NUMBER OF ATOMS \n")
    file.write("%i \n" % natoms)

    file.write("ITEM: BOX BOUNDS \n")
    file.write("%e %e \n" % (settings.bounds[0,0], settings.bounds[0,1]))
    file.write("%e %e \n" % (settings.bounds[1,0], settings.bounds[1,1]))
    file.write("%e %e \n" % (settings.bounds[2,0], settings.bounds[2,1]))
    # file.write("%e %e \n" % (settings.bounds[0,0]*settings.conv_length, settings.bounds[0,1]*settings.conv_length))
    # file.write("%e %e \n" % (settings.bounds[1,0]*settings.conv_length, settings.bounds[1,1]*settings.conv_length))
    # file.write("%e %e \n" % (settings.bounds[2,0]*settings.conv_length, settings.bounds[2,1]*settings.conv_length))

    if natoms%3 == 0:
        file.write("ITEM: ATOMS id mol type x y z vx vy vz fx fy fz mass\n")
        for i in range(natoms):
            itype = "O" if i%3 == 0 else "H"
            file.write("%i %i %s %e %e %e %e %e %e %e %e %e %e\n"
                % (i, i//3, itype, x[i], y[i], z[i], vx[i], vy[i], vz[i], fx[i], fy[i], fz[i], masses[i]))
            # file.write(
            #     "%i %i %s %e %e %e %e %e %e %e %e %e %e\n"
            #     % (i, i//3, itype,
            #        x[i]*settings.conv_length, y[i]*settings.conv_length, z[i]*settings.conv_length,
            #        vx[i]*settings.conv_velocity, vy[i]*settings.conv_velocity, vz[i]*settings.conv_velocity,
            #        fx[i]*settings.conv_force, fy[i]*settings.conv_force, fz[i]*settings.conv_force,
            #        settings.masses[i]*settings.conv_mass)
            # )
    else:
        file.write("ITEM: ATOMS id mol type x y z vx vy vz fx fy fz mass\n")
        for i in range(natoms):
            if i < natoms-2:
                itype = "O" if i%3 == 0 else "H"
            else:
                itype = "Na" if i == natoms-2 else "I"
            file.write("%i %i %s %e %e %e %e %e %e %e %e %e %e\n"
                % (i, i//3, itype, x[i], y[i], z[i], vx[i], vy[i], vz[i], fx[i], fy[i], fz[i], masses[i]))