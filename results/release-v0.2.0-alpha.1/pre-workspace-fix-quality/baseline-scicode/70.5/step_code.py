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