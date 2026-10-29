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
def reciprocal(*arg):
    '''Computes the reciprocal vector(s) based on the input vector(s).
    Input:
    *arg (numpy array(s)): some vectors, with the amount uncertain within the range of [1, 2, 3]
    Output:
    rec (list of numpy array(s)): The collection of all the reciprocal vectors.
    '''
    n = len(arg)
    if n == 1:
        a1 = np.asarray(arg[0]).flatten()
        denom = np.dot(a1, a1)
        b1 = 2 * np.pi * a1 / denom
        return [b1]
    elif n == 2:
        a1 = np.asarray(arg[0]).flatten()
        a2 = np.asarray(arg[1]).flatten()
        a1_3d = np.append(a1, 0.0)
        a2_3d = np.append(a2, 0.0)
        a3 = np.array([0.0, 0.0, 1.0])
        rec3d = reciprocal_3D(a1_3d, a2_3d, a3)
        b1 = rec3d[0][:2]
        b2 = rec3d[1][:2]
        return [b1, b2]
    elif n == 3:
        flattened = [np.asarray(a).flatten() for a in arg]
        return reciprocal_3D(*flattened)
    else:
        return []