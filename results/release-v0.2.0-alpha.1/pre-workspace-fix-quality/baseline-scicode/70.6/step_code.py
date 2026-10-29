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