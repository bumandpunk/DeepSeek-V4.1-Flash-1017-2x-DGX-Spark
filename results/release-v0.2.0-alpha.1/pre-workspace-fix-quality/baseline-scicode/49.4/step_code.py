def Nbody(uin, mass, dt, t0, t1):
    '''This function runs n-body simulation for N particles interacting via gravity only.
    Inputs:
    uin: a 2D array of initial positions and velocities. N,6 elements; each element is a float; each particle is described by its location and velocity, first the $x$,$y$,$z$ position of the particle followed by the $vx$,$vy$,$vz$ velocity of the partile. Thus particle "i"'s position and velocity are stored in uin[i][0],uin[i][1],uin[i][2],uin[i][3],uin[i][4],uin[i][5].
    mass: an 1D array of particle masses; an array of floats
    dt: time step size; a float
    t0: initial time; a float
    t1: final time; a float
    Outputs:
    out: final positions and velocities of particles; an array of floats laid out like uin
    '''
    uout = np.asarray(uin, dtype=float).copy()
    t = float(t0)
    tf = float(t1)
    h = float(dt)

    if h == 0.0:
        if t == tf:
            return uout
        raise ValueError("dt must be nonzero")

    eps = 1e-12 * max(1.0, abs(tf), abs(t0), abs(h))

    if h > 0.0:
        while t < tf - eps:
            step = min(h, tf - t)
            uout = Nbody_RK4(uout, mass, step)
            t += step
        if abs(tf - t) > eps:
            uout = Nbody_RK4(uout, mass, tf - t)
    else:
        while t > tf + eps:
            step = max(h, tf - t)
            uout = Nbody_RK4(uout, mass, step)
            t += step
        if abs(tf - t) > eps:
            uout = Nbody_RK4(uout, mass, tf - t)

    return uout