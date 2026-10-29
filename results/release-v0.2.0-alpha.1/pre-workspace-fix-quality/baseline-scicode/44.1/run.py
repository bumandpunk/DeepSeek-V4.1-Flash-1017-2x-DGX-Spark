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
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('44.1',3,"/data/test_data.h5")

target=targets[0]
d = np.array([[1, 2], [3, 4]])
assert np.allclose(GetComp(d), target)

target=targets[1]
d = np.array([[0.9204365 , 0.99516558, 0.76198302, 0.50127276],
       [0.78900999, 0.53480707, 0.68065993, 0.20993823],
       [0.32379346, 0.29234994, 1.20530971, 0.6083708 ],
       [1.33007112, 0.37400537, 5.03614455, 0.79329857]])
assert np.allclose(GetComp(d), target)

target=targets[2]
d = np.array([[0.7513931 , 0.38820798, 1.5799524 , 4.81004798, 0.56173512,
        1.09404242],
       [0.68588083, 1.77330067, 1.29963916, 0.34003408, 0.84649143,
        0.30521963],
       [1.22523679, 0.16233243, 0.82777415, 0.29361007, 1.23801451,
        0.31543167],
       [0.41923618, 1.49125219, 0.5071743 , 0.35692214, 0.24710065,
        0.10839822],
       [4.51486227, 0.51183228, 3.16783014, 1.05718153, 0.28319028,
        1.14236583],
       [0.47367523, 1.15941838, 0.47917275, 2.35092014, 1.30285756,
        0.29673001]])
assert np.allclose(GetComp(d), target)
