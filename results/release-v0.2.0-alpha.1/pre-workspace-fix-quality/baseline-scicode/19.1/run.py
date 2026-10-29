import numpy as np
from scipy.linalg import sqrtm

def tensor(*args):
    '''Takes the tensor product of an arbitrary number of matrices/vectors.
    Input:
    args: any number of arrays, corresponding to input matrices
    Output:
    M: the tensor product (kronecker product) of input matrices
    '''
    if len(args) == 0:
        return np.array(1.0)

    M = args[0]
    for matrix in args[1:]:
        M = np.kron(M, matrix)

    return M
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('19.1',3,"/data/test_data.h5")

target=targets[0]
assert np.allclose(tensor([0,1],[0,1]), target)

target=targets[1]
assert np.allclose(tensor(np.eye(3),np.ones((3,3))), target)

target=targets[2]
assert np.allclose(tensor([[1/2,1/2],[0,1]],[[1,2],[3,4]]), target)
