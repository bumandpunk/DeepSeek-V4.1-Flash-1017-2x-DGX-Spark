import numpy as np
from scipy.spatial.distance import pdist

def U_ij(r, sigma, epsilon):
    '''Lennard Jones Potential between pair of atoms with distance r
    Inputs:
    r: distance, float
    sigma: the distance at which Lennard Jones potential reaches zero, float
    epsilon: potential well depth of Lennard Jones potential, float
    Outputs:
    U: Potential Energy, float
    '''

    ratio = sigma / r
    U = 4.0 * epsilon * (ratio**12 - ratio**6)
    return U