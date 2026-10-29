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
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('44.2',3,"/data/test_data.h5")

target=targets[0]
t = 0
y = np.array([[1, 2], [3, 4]]).flatten()
c = np.ones(2)*10
lig = np.array([[1, 0], [0, 1]])
assert np.allclose(step(t, y, c, lig), target)

target=targets[1]
t = 0
y = np.array([[0.9204365 , 0.99516558, 0.76198302, 0.50127276],
       [0.78900999, 0.53480707, 0.68065993, 0.20993823],
       [0.32379346, 0.29234994, 1.20530971, 0.6083708 ],
       [1.33007112, 0.37400537, 5.03614455, 0.79329857]]).flatten()
c = np.ones(4)*10
lig = np.array([[0.8997858 , 1.01790812, 1.23096801, 0.59830888],
       [1.34481097, 1.33982066, 0.72675653, 0.78848685],
       [0.79413525, 0.97270199, 0.96545486, 0.80494749],
       [0.87638968, 1.09389482, 0.75718173, 1.09107295]])
assert np.allclose(step(t, y, c, lig), target)

target=targets[2]
t = 20
y = np.array([[0.9204365 , 0.99516558, 0.76198302, 0.50127276],
       [0.78900999, 0.53480707, 0.68065993, 0.20993823],
       [0.32379346, 0.29234994, 1.20530971, 0.6083708 ],
       [1.33007112, 0.37400537, 5.03614455, 0.79329857]]).flatten()
c = np.ones(4)*10
lig = np.eye(4)
assert np.allclose(step(t, y, c, lig), target)
