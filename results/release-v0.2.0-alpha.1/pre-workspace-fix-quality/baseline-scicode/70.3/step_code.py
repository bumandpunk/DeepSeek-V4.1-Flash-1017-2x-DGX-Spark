def hamiltonian_3nu_su3_coefficients(hamiltonian):
    '''Returns the h_k of the SU(3)-expansion of the 3nu Hamiltonian.
    Input
    hamiltonian: a list of lists containing the 3x3 Hamiltonian matrix; each inner list contains three complex numbers
    Output
    hks: a list containing the h_k coefficients of the 3nu Hamiltonian in the expansion using the Gell-Mann matrices (k=1 to k=8); a list of floats (possibly complex)
    '''

    H = hamiltonian

    h1 = H[0][1].real
    h2 = -H[0][1].imag
    h3 = 0.5 * (H[0][0].real - H[1][1].real)
    h4 = H[0][2].real
    h5 = -H[0][2].imag
    h6 = H[1][2].real
    h7 = -H[1][2].imag
    h8 = (np.sqrt(3.0) / 6.0) * (H[0][0].real + H[1][1].real - 2.0 * H[2][2].real)

    return [h1, h2, h3, h4, h5, h6, h7, h8]