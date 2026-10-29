import numpy as np

def cross(a, b):
    '''Calculates the cross product of the input vectors.
    Input:
    a (numpy array): Vector a.
    b (numpy array): Vector b.
    Output:
    t (numpy array): The cross product of a and b.
    '''
    t = np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ])
    return t
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('38.1',3,"/data/test_data.h5")

target=targets[0]
a,b = np.array([3,4,5]),np.array([4,3,2])
assert np.allclose(cross(a,b), target)

target=targets[1]
a,b = np.array([3,4,7]),np.array([8,2,6])
assert np.allclose(cross(a,b), target)

target=targets[2]
a,b = np.array([1,1,0]),np.array([1,-1,0])
assert np.allclose(cross(a,b), target)
