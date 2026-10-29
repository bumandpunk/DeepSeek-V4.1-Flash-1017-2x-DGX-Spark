import numpy as np
def normalize(v):
    '''Normalize the input vector.
    Input:
    v (N*1 numpy array): The input vector.
    Output:
    n (N*1 numpy array): The normalized vector.
    '''
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.copy()
    n = v / norm
    return n
def inner_product(u, v):
    '''Calculates the inner product of two vectors.
    Input:
    u (numpy array): Vector 1.
    v (numpy array): Vector 2.
    Output:
    p (float): Inner product of the vectors.
    '''
    p = float(np.sum(u * v))
    return p
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('29.2',3,"/data/test_data.h5")

target=targets[0]
u,v = np.array([3,4]),np.array([4,3])
assert np.allclose(inner_product(u,v), target)

target=targets[1]
u,v = np.array([3,4]),np.array([3,4])
assert np.allclose(inner_product(u,v), target)

target=targets[2]
u,v = np.array([3,4,7,6]),np.array([4,3,2,8])
assert np.allclose(inner_product(u,v), target)
