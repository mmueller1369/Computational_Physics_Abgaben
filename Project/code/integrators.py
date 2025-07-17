import settings
settings.init()


def VelocityVerlet(
    x, y, z,
    vx, vy, vz,
    fx, fy, fz,
    masses,
    dt,
    force_func, force_params
    ):

    # update the position at t+dt
    x += vx*dt + fx/settings.conv_factor*dt*dt/2/masses
    y += vy*dt + fy/settings.conv_factor*dt*dt/2/masses
    z += vz*dt + fz/settings.conv_factor*dt*dt/2/masses

    # save the force at t
    fx0 = fx
    fy0 = fy
    fz0 = fz
    # update acceleration at t+dt
    fx, fy, fz, energies = force_func(x, y, z, *force_params)
    
    # update the velocity
    vx += dt/masses/2 * (fx+fx0)/settings.conv_factor
    vy += dt/masses/2 * (fy+fy0)/settings.conv_factor
    vz += dt/masses/2 * (fz+fz0)/settings.conv_factor

    return x, y, z, vx, vy, vz, fx, fy, fz, energies