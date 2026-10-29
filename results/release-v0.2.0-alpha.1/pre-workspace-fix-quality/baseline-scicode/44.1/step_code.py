def GetComp(d):
    '''Concentration matrix of complementary 2-mers
    Inputs:
    d: concentration of 2-mers, numpy float array with dimensions [Z, Z]
    Outputs:
    dcomp: concentration of 2-mers, numpy float array with dimensions [Z, Z], where dcomp[i, j]==d[j', i'].
    '''
    Z = d.shape[0]
    half = Z // 2
    indices = np.arange(Z)
    comp = np.where(indices < half, indices + half, indices - half)
    dcomp = d[np.ix_(comp, comp)].T

    return dcomp