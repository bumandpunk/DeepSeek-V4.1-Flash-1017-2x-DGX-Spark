def distance_matrix(configs):
    '''Args:
        configs (np.array): (nparticles, 3)
    Returns:
        distances (np.array): distance vector for each particle pair. Shape (npairs, 3), where npairs = (nparticles choose 2)
        pair_idxs (list of tuples): list of pair indices
    '''
    nparticles = len(configs)
    idx_i, idx_j = np.triu_indices(nparticles, k=1)
    pair_idxs = list(zip(idx_i.tolist(), idx_j.tolist()))
    distances = configs[idx_i] - configs[idx_j]
    return distances, pair_idxs