def Nbody_RK4(uin, mass, dt):
    '''This function takes a single step of size $dt$ computing an updated state vector $uout$ given the current state $uin$, $mass$, and timestep $dt$.
    Inputs:
    uin: a 2D array of current positions and velocities. N,6 elements; each element is a float; each particle is described by its location and velocity, first the $x$,$y$,$z$ position of the particle followed by the $vx$,$vy$,$vz$ velocity of the partile. Thus particle "i"'s position and velocity are stored in uin[i][0],uin[i][1],uin[i][2],uin[i][3],uin[i][4],uin[i][5].
    mass: an 1D array of particle masses; an array of floats
    dt: time step to use for the evolution; a float
    Outputs:
    uout: a 2D array of final positions and velocities, N,6 elements; each element is a float; laid out like uin
    '''
    uin_array = np.asarray(uin, dtype=float)
    original_shape = uin_array.shape

    if uin_array.ndim == 1:
        mass_array = np.asarray(mass, dtype=float)
        N = mass_array.size
        u = uin_array.reshape((N, 6))
    else:
        u = uin_array

    k1 = Nbody_RHS(u, mass)
    k2 = Nbody_RHS(u + 0.5 * dt * k1, mass)
    k3 = Nbody_RHS(u + 0.5 * dt * k2, mass)
    k4 = Nbody_RHS(u + dt * k3, mass)

    uout = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    if len(original_shape) == 1:
        return uout.reshape(-1)

    return uout