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
def reciprocal_3D(a1, a2, a3):
    '''Calculates the 3D reciprocal vectors given the input.
    Input:
    a1 (numpy array): Vector 1.
    a2 (numpy array): Vector 2.
    a3 (numpy array): Vector 3.
    Returns:
    b_i (list of numpy arrays): The collection of reciprocal vectors.
    '''
    denom = np.dot(a1, cross(a2, a3))
    b_i = [
        2 * np.pi * cross(a2, a3) / denom,
        2 * np.pi * cross(a3, a1) / denom,
        2 * np.pi * cross(a1, a2) / denom
    ]
    return b_i
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('38.2',3,"/data/test_data.h5")

target=targets[0]
a1,a2,a3 = np.array([3,4,5]),np.array([4,3,2]),np.array([1,0,2])
assert np.allclose(reciprocal_3D(a1,a2,a3), target)

target=targets[1]
a1,a2,a3 = np.array([3,4,7]),np.array([8,2,6]),np.array([1,0,2])
assert np.allclose(reciprocal_3D(a1,a2,a3), target)

target=targets[2]
a1,a2,a3 = np.array([1,1,0]),np.array([1,-1,0]),np.array([1,0,2])
assert np.allclose(reciprocal_3D(a1,a2,a3), target)
