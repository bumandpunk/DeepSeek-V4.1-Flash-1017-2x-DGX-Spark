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
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('44.3',3,"/data/test_data.h5")

target=targets[0]
c0 = np.ones(4)*10.0
d0 = np.ones([4, 4])*0.5
lig = np.hstack([np.ones([4, 2]), -np.ones([4, 2])])
tf = 20
nsteps = 1000
assert np.allclose(entropies(c0, d0, lig, tf, nsteps), target)

target=targets[1]
c0 = np.ones(8)
d0 = np.ones([8, 8])*0.01
lig = np.array([[1.21663772, 0.89248457, 1.21745463, 1.11149841, 0.81367134,
        0.89963613, 1.30729711, 1.12344434],
       [0.7829948 , 0.97670401, 1.17241282, 1.17666507, 0.99661672,
        0.82597011, 1.26869751, 0.92559714],
       [1.32370699, 0.81555449, 1.24539873, 1.11920104, 0.70893291,
        0.94277028, 0.99222404, 1.28004994],
       [1.33605707, 0.9677957 , 0.87545897, 1.11743411, 1.00387065,
        1.21387094, 1.3587547 , 1.07888797],
       [1.09560243, 1.22427195, 0.91200443, 0.79503969, 1.32678448,
        1.0089972 , 1.40026918, 1.1007811 ],
       [1.07280747, 1.10334304, 0.69314364, 1.47071329, 1.20877084,
        0.70031823, 0.90295041, 0.8843919 ],
       [0.89329015, 0.97337332, 1.16629781, 1.00359901, 1.05891634,
        0.68708136, 1.26150524, 0.87881251],
       [1.47199815, 0.69658061, 0.84842985, 1.22751336, 1.01383103,
        0.99759963, 0.69677971, 0.99616318]])
tf = 1000
nsteps = 10000
assert np.allclose(entropies(c0, d0, lig, tf, nsteps), target)

target=targets[2]
c0 = np.array([7.0681359 , 2.62361004, 1.59922841, 4.01258308, 0.89411538,
       2.94376844, 1.85685374, 3.45821068])
d0 = np.ones([8, 8])*0.01
lig = np.array([[0.77124931, 0.95367699, 0.98325401, 1.08071122, 1.19929357,
        1.91809708, 0.71432156, 1.27096494],
       [0.85762676, 1.0169952 , 1.18012769, 1.33453853, 1.391186  ,
        1.08892087, 0.96188594, 1.01095109],
       [1.0255774 , 0.97554961, 0.94428129, 0.64227944, 0.73790089,
        0.81056697, 0.91220593, 1.54502225],
       [0.8066888 , 0.81817817, 0.91694827, 1.00399813, 0.82554822,
        0.92915596, 0.83144055, 1.40705141],
       [0.7386233 , 0.82211199, 1.2920814 , 1.1757341 , 1.06815192,
        0.84285394, 1.71679303, 0.60512826],
       [1.16674491, 1.07334837, 1.0456343 , 0.85723942, 1.33040205,
        0.80796219, 0.73944967, 1.21658198],
       [1.03109573, 1.07236791, 0.94173481, 1.17093906, 0.77731087,
        1.02610659, 0.78401785, 0.82595097],
       [1.40816645, 1.12174143, 0.89368929, 1.10921353, 1.06705567,
        0.78623361, 0.97364678, 1.05773036]])
tf = 1000
nsteps = 10000
assert np.allclose(entropies(c0, d0, lig, tf, nsteps), target)
