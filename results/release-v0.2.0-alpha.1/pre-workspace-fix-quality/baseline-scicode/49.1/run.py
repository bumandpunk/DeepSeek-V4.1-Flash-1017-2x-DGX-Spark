import numpy as np

import numpy as np

def Nbody_Fij(xin, mass):
    '''This function computes the forces between particles subject to gravity.
    Inputs:
    xin: 2D array of locations for the N particles, N,3 elements where 3 is x,y,z coordinates
    mass: 1D array of particle masses, N elements
    Outputs:
    f: 2D array of forces, on each particle, N,3 elements where 3 is x,y,z coordinates
    '''
    mass = np.asarray(mass, dtype=float)
    x = np.asarray(xin, dtype=float)
    original_shape = x.shape

    if x.ndim == 1:
        N = mass.size
        x = x.reshape((N, 3))
    elif x.ndim == 2:
        N = x.shape[0]
    else:
        raise ValueError("xin must be a 1D array of length 3*N or a 2D array of shape (N, 3)")

    f = np.zeros((N, 3), dtype=float)
    G = 6.67430e-11

    for i in range(N):
        rij = x - x[i]
        r2 = np.sum(rij * rij, axis=1)
        r2[i] = np.inf

        inv_r3 = np.zeros(N, dtype=float)
        mask = r2 > 0.0
        inv_r3[mask] = 1.0 / (r2[mask] * np.sqrt(r2[mask]))

        f[i] = G * mass[i] * np.sum((mass * inv_r3)[:, None] * rij, axis=0)

    if len(original_shape) == 1:
        return f.reshape(-1)
    return f
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('49.1',3,"/data/test_data.h5")

target=targets[0]
xin = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
mass = np.array([1.0e4, 1.0e4])
assert np.allclose(Nbody_Fij(xin, mass), target)

target=targets[1]
xin = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(0.75), 0.0]])
mass = np.array([1.0e3, 1.5e4, 2.0e5])
assert np.allclose(Nbody_Fij(xin, mass), target)

target=targets[2]
xin = np.array([[0.89660964, 0.20404593, 0.26538925],
       [0.83776986, 0.60298593, 0.26242915],
       [0.47325228, 0.77652396, 0.4536616 ],
       [0.11320759, 0.9165638 , 0.8266307 ],
       [0.4766697 , 0.56786154, 0.0529977 ]])
mass = np.array([0.72386384e7, 0.73792416e3, 0.78105276e2, 0.30785728e5, 0.59321658e6])
assert np.allclose(Nbody_Fij(xin, mass), target)
