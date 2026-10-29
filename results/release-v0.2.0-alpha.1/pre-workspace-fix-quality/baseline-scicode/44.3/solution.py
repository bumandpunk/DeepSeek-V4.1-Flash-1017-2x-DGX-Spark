import numpy as np
from math import exp
from scipy.integrate import solve_ivp
def GetComp(d):
    '''Concentration matrix of complementary 2-mers
    Inputs:
    d: concentration of 2-mers, numpy float array with dimensions [Z, Z]
    Outputs:
    dcomp: concentration of 2-mers, numpy float array with dimensions [Z, Z], where dcomp[i, j]==d[j', i'].
    '''
    Z = d.shape[0]
    half = Z // 2
    indices = np.arange(Z)
    comp = np.where(indices < half, indices + half, indices - half)
    dcomp = d[np.ix_(comp, comp)].T

    return dcomp
def step(t, y, c, lig):
    '''To integrate the master equation of all the 2-mers
    Inputs:
    t: timepoint, float
    y: flattened current state of the d matrix, numpy float array with length [Z^2]
    c: concentration of monomers of each type, numpy float array with length [Z]
    lig: lambda_{ij} matrix of each 2-mer's ligation rate during the night phase, numpy float array with dimensions [Z, Z]
    Outputs:
    ystep: flattened changing rate of the d matrix, numpy float array with length [Z^2]
    '''
    Z = c.shape[0]
    d = y.reshape((Z, Z))

    half = Z // 2
    indices = np.arange(Z)
    comp = np.where(indices < half, indices + half, indices - half)
    dcomp = d[np.ix_(comp, comp)].T

    r = c - np.sum(d, axis=1)
    l = c - np.sum(d, axis=0)

    ystep = (lig * r[:, np.newaxis] * l[np.newaxis, :] * dcomp - d).reshape(-1)
    return ystep
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