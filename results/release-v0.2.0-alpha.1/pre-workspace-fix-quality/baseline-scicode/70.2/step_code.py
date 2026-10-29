def hamiltonian_3nu(s12, s23, s13, dCP, D21, D31):
    U = pmns_mixing_matrix(s12, s23, s13, dCP)
    D = np.diag(np.array([0.0, D21, D31], dtype=complex))
    H = 0.5 * U @ D @ np.conjugate(U.T)
    return [[complex(H[i, j]) for j in range(3)] for i in range(3)]