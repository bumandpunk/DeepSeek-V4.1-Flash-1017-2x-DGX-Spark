import numpy as np
import cmath 
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
def hamiltonian_3nu(s12, s23, s13, dCP, D21, D31):
    U = pmns_mixing_matrix(s12, s23, s13, dCP)
    D = np.diag(np.array([0.0, D21, D31], dtype=complex))
    H = 0.5 * U @ D @ np.conjugate(U.T)
    return [[complex(H[i, j]) for j in range(3)] for i in range(3)]
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
def tensor_d(i, j, k):
    '''Returns the tensor d_ijk of the SU(3) algebra.
    Input
    i: the first index; int
    j: the second index; int
    k: the third index; int
    Output
    result: the value of d_ijk for given indices i, j, and k; float
    '''

    def gell_mann(n):
        if n == 1:
            return np.array([[0, 1, 0],
                             [1, 0, 0],
                             [0, 0, 0]], dtype=complex)
        elif n == 2:
            return np.array([[0, -1j, 0],
                             [1j, 0, 0],
                             [0, 0, 0]], dtype=complex)
        elif n == 3:
            return np.array([[1, 0, 0],
                             [0, -1, 0],
                             [0, 0, 0]], dtype=complex)
        elif n == 4:
            return np.array([[0, 0, 1],
                             [0, 0, 0],
                             [1, 0, 0]], dtype=complex)
        elif n == 5:
            return np.array([[0, 0, -1j],
                             [0, 0, 0],
                             [1j, 0, 0]], dtype=complex)
        elif n == 6:
            return np.array([[0, 0, 0],
                             [0, 0, 1],
                             [0, 1, 0]], dtype=complex)
        elif n == 7:
            return np.array([[0, 0, 0],
                             [0, 0, -1j],
                             [0, 1j, 0]], dtype=complex)
        elif n == 8:
            sqrt3 = 1.0 / np.sqrt(3.0)
            return np.array([[sqrt3, 0, 0],
                             [0, sqrt3, 0],
                             [0, 0, -2.0 * sqrt3]], dtype=complex)
        else:
            return np.zeros((3, 3), dtype=complex)

    lambda_i = gell_mann(i)
    lambda_j = gell_mann(j)
    lambda_k = gell_mann(k)

    anticommutator = lambda_i @ lambda_j + lambda_j @ lambda_i
    result = np.real(np.trace(anticommutator @ lambda_k)) / 4.0

    return float(result)
def star_product(i, h):
    '''Returns the SU(3) star product (h*h)_i = d_ijk*h^j*h^k (summed over
    repeated indices).
    Input
    i: index of the star product; int
    h: a list of eight expansion coefficients; a list of floats (possibly complex)
    Output
    product: the star product (h*h)_i; complex numbers
    '''

    product = 0j

    for j in range(1, 9):
        for k in range(1, 9):
            product += tensor_d(i, j, k) * h[j - 1] * h[k - 1]

    return complex(product)
def psi_m(h):
    '''Input
    h: a list of eight expansion coefficients; a list of floats (possibly complex)
    Output
    result: a list containing the following three variables:
            h2: SU(3) invariant |h|^2; complex number
            h3: SU(3) invariant <h>; complex number
            psi_list: a list of three complex number
    '''

    h2 = sum(h[i - 1] * h[i - 1] for i in range(1, 9))
    h3 = sum(h[i - 1] * star_product(i, h) for i in range(1, 9))

    h2 = complex(h2)
    h3 = complex(h3)

    if h2 == 0j:
        psi_list = [0j, 0j, 0j]
    else:
        abs_h = cmath.sqrt(h2)
        if abs_h == 0j:
            psi_list = [0j, 0j, 0j]
        else:
            cos_chi = -np.sqrt(3.0) * h3 / (abs_h ** 3)
            chi = cmath.acos(cos_chi)

            psi_list = []
            for m in range(1, 4):
                psi = (2.0 * abs_h / np.sqrt(3.0)) * cmath.cos(
                    (chi + 2.0 * np.pi * m) / 3.0
                )
                psi_list.append(complex(psi))

    return [h2, h3, psi_list]
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('70.6',3,"/data/test_data.h5")

target=targets[0]
from scicode.compare.cmp import cmp_tuple_or_list
s12 = 1/cmath.sqrt(3)
s13 = 1
s23 = 1/cmath.sqrt(2)
dCP = 0
D21 = 1e-4
D31 = 1e-3
hamiltonian = hamiltonian_3nu(s12, s13, s23, dCP, D21, D31)
h_coefficient = hamiltonian_3nu_su3_coefficients(hamiltonian)
assert cmp_tuple_or_list(psi_m(h_coefficient), target)

target=targets[1]
from scicode.compare.cmp import cmp_tuple_or_list
s12 = 1/cmath.sqrt(2)
s13 = 0
s23 = 1/cmath.sqrt(2)
dCP = 0
D21 = 1e-4
D31 = 1e-3
hamiltonian = hamiltonian_3nu(s12, s13, s23, dCP, D21, D31)
h_coefficient = hamiltonian_3nu_su3_coefficients(hamiltonian)
assert cmp_tuple_or_list(psi_m(h_coefficient), target)

target=targets[2]
from scicode.compare.cmp import cmp_tuple_or_list
s12 = 1/cmath.sqrt(2)
s13 = 0
s23 = 1/cmath.sqrt(3)
dCP = 1
D21 = 5e-4
D31 = 5e-3
hamiltonian = hamiltonian_3nu(s12, s13, s23, dCP, D21, D31)
h_coefficient = hamiltonian_3nu_su3_coefficients(hamiltonian)
assert cmp_tuple_or_list(psi_m(h_coefficient), target)
