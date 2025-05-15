def g_r(x, y, z, xlo, xhi, ylo, yhi, zlo, zhi):
    """
    Calculate the g(r) function for a given set of coordinates.
    """
    # Initialize variables
    N = len(x)
    dr = 0.1  # in sigma units
    rmax = settings.xhi
    nbins = int(rmax / dr)
    g_r = np.zeros(nbins)
    rho = N / ((xhi - xlo) * (yhi - ylo) * (zhi - zlo))
    V = (xhi - xlo) * (yhi - ylo) * (zhi - zlo)

    # Loop over all pairs of particles
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate the distance between particles i and j
            dx = pbc(x[i], x[j], xlo, xhi)
            dy = pbc(y[i], y[j], ylo, yhi)
            dz = pbc(z[i], z[j], zlo, zhi)
            r2 = dx**2 + dy**2 + dz**2
            r = math.sqrt(r2)

            # Bin the distance
            if r < rmax:
                bin_index = int(r / dr)
                g_r[bin_index] += 2.0 / (rho * V * dr)

    return g_r
