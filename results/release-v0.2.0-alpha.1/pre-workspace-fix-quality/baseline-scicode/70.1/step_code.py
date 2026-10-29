def pmns_mixing_matrix(s12, s23, s13, dCP):
    '''Returns the 3x3 PMNS mixing matrix.
    Computes and returns the 3x3 complex PMNS mixing matrix
    parameterized by three rotation angles: theta_12, theta_23, theta_13,
    and one CP-violation phase, delta_CP.
    Input
    s12 : Sin(theta_12); float
    s23 : Sin(theta_23); float
    s13 : Sin(theta_13); float
    dCP : delta_CP in radians; float
    Output
    pmns: numpy array of floats with shape (3,3) containing the 3x3 PMNS mixing matrix
    '''

    c12 = np.sqrt(1.0 - s12**2)
    c23 = np.sqrt(1.0 - s23**2)
    c13 = np.sqrt(1.0 - s13**2)

    phase_plus = cmath.exp(1j * dCP)
    phase_minus = cmath.exp(-1j * dCP)

    pmns = np.array([
        [c12 * c13, s12 * c13, s13 * phase_minus],
        [-s12 * c23 - c12 * s23 * s13 * phase_plus,
         c12 * c23 - s12 * s23 * s13 * phase_plus,
         s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * phase_plus,
         -c12 * s23 - s12 * c23 * s13 * phase_plus,
         c23 * c13]
    ], dtype=complex)

    return pmns