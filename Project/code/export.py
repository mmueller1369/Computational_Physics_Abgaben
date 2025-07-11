import settings
# import settings_SI as settings
import tools
settings.init()


def WriteEnergy(file, time, vx, vy, vz, e_LJ, e_coul, e_bond, e_angle):
    ekin = tools.computeKineticEnergy(vx, vy, vz, settings.masses)
    file.write("%i %e %e %e %e %e\n" % (time,
        e_LJ, e_coul, e_bond, e_angle, ekin))
    # file.write("%i %e %e %e %e %e\n" % (time,
    #     e_LJ*settings.conv_energy, e_coul*settings.conv_energy, e_bond*settings.conv_energy, e_angle*settings.conv_energy, ekin*settings.conv_energy))


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
    # file.write("%e %e \n" % (settings.bounds[0,0]*settings.conv_length, settings.bounds[0,1]*settings.conv_length))
    # file.write("%e %e \n" % (settings.bounds[1,0]*settings.conv_length, settings.bounds[1,1]*settings.conv_length))
    # file.write("%e %e \n" % (settings.bounds[2,0]*settings.conv_length, settings.bounds[2,1]*settings.conv_length))

    file.write("ITEM: ATOMS id mol type x y z vx vy vz fx fy fz mass\n")
    for i in range(natoms):
        itype = "O" if i%3 == 0 else "H"
        file.write(
            "%i %i %s %e %e %e %e %e %e %e %e %e %e\n"
            % (i, i//3, itype, x[i], y[i], z[i], vx[i], vy[i], vz[i], fx[i], fy[i], fz[i], settings.masses[i])
        )
        # file.write(
        #     "%i %i %s %e %e %e %e %e %e %e %e %e %e\n"
        #     % (i, i//3, itype,
        #        x[i]*settings.conv_length, y[i]*settings.conv_length, z[i]*settings.conv_length,
        #        vx[i]*settings.conv_velocity, vy[i]*settings.conv_velocity, vz[i]*settings.conv_velocity,
        #        fx[i]*settings.conv_force, fy[i]*settings.conv_force, fz[i]*settings.conv_force,
        #        settings.masses[i]*settings.conv_mass)
        # )
    # for i in range(natoms):
    #     if i%3 == 0:
    #         file.write(
    #             "%i %i %s %e %e %e %e %e %e %e %e %e %e\n"
    #             % (i, i//3, "O", x[i], y[i], z[i], vx[i], vy[i], vz[i], fx[i], fy[i], fz[i], settings.masses[i])
    #         )
    #     else:
    #         file.write(
    #             "%i %i %s %e %e %e %e %e %e %e %e %e %e\n"
    #             % (i, i//3, "H", x[i], y[i], z[i], 0, 0, 0, fx[i], fy[i], fz[i], settings.masses[i])
    #     )
        