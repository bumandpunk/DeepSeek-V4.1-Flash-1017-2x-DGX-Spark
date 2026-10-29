def select_big_weights(gpoints_all, cell_volume, alpha, tol=1e-10):
    '''Filter g-points based on weight in reciprocal space.
    Parameters:
        gpoints_all (np.ndarray): An array of shape (nk, 3) representing all g-points.
        cell_volume (float): The volume of the unit cell.
        alpha (float): The alpha value used for the Ewald summation.
        tol (float, optional): The tolerance for filtering weights. Default is 1e-10.
    Returns:
        tuple: 
            gpoints (np.array): An array of shape (nk, 3) containing g-points with significant weights.
            gweights (np.array): An array of shape (nk,) containing the weights of the selected g-points.       
    '''
    gpoints_all = np.asarray(gpoints_all)
    k2 = np.sum(gpoints_all**2, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        gweights_all = (4.0 * np.pi / cell_volume) * np.exp(-k2 / (4.0 * alpha**2)) / k2

    mask = (k2 > 0.0) & (gweights_all > tol)

    gpoints = gpoints_all[mask]
    gweights = gweights_all[mask]

    return gpoints, gweights