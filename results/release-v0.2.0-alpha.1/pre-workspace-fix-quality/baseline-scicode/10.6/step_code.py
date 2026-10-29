def generate_gpoints(recvec, gmax):
    '''Generate a grid of g-points for reciprocal space based on the provided lattice vectors.
    Parameters:
        recvec (np.ndarray): A 3x3 array representing the reciprocal lattice vectors.
        gmax (int): The maximum integer number of lattice points to include in one positive direction.
    Returns:
        np.ndarray: An array of shape (nk, 3) representing the grid of g-points.
    '''
    x_vals = np.arange(0, gmax + 1)
    yz_vals = np.arange(-gmax, gmax + 1)

    grid = np.meshgrid(x_vals, yz_vals, yz_vals, indexing='ij')
    gpoints = np.stack(grid, axis=-1).reshape(-1, 3)

    x = gpoints[:, 0]
    y = gpoints[:, 1]
    z = gpoints[:, 2]

    valid = (
        (x > 0)
        | ((x == 0) & (y > 0))
        | ((x == 0) & (y == 0) & (z > 0))
    )

    gpoints_all = gpoints[valid] @ recvec
    return gpoints_all