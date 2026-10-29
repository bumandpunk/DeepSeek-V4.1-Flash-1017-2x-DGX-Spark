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
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('29.1',3,"/data/test_data.h5")

target=targets[0]
v = np.array([3,4])
assert np.allclose(normalize(v), target)

target=targets[1]
v = np.array([1,2,3,4,5,6,7,8]).reshape(4,2)
assert np.allclose(normalize(v), target)

target=targets[2]
v = np.array([i for i in range(12)]).reshape(3,2,2)
assert np.allclose(normalize(v), target)
