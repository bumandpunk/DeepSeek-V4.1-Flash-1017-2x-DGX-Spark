def entropies(c0, d0, lig, tf, nsteps):
    '''To calculate the entropy of the system at each timestep
    Inputs:
    c0: concentration of monomers of each type at time 0, numpy float array with length [Z]
    d0: concentration of 2-mers of each type at time 0, numpy float array with dimensions [Z, Z]
    lig: lambda_{ij} matrix of each 2-mer's ligation rate during the night phase, numpy float array with dimensions [Z, Z]
    tf: the simulation goes from 0 to tf, float
    nsteps: the number of simulation steps, int
    Outputs:
    entropy_list: entropy at each timestep, numpy array of length nsteps
    '''
    Z = c0.shape[0]
    t_eval = np.linspace(0.0, tf, nsteps)
    y0 = np.asarray(d0, dtype=float).reshape(-1)

    def rhs(t, y):
        return step(t, y, c0, lig)

    sol = solve_ivp(rhs, (0.0, tf), y0, t_eval=t_eval)
    D = sol.y.T.reshape((nsteps, Z, Z))

    totals = np.sum(D, axis=(1, 2))
    denom = totals[:, np.newaxis, np.newaxis]

    dtilde = np.divide(
        D,
        denom,
        out=np.zeros_like(D, dtype=float),
        where=denom != 0
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        plogp = np.where(dtilde > 0, dtilde * np.log(dtilde), 0.0)

    entropy_list = -np.sum(plogp, axis=(1, 2))
    return entropy_list