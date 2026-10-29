def apply_band_pass_filter(image_array, bandmin, bandmax):
    '''Applies a band pass filter to the given image array based on the frequency threshold.
    Input:
    image_array: 2D numpy array of float, the input image.
    bandmin: float, inner radius of the frequenc band
    bandmax: float, outer radius of the frequenc band
    Ouput:
    T: float,2D numpy array, The spatial filter used.
    output_image: float,2D numpy array, the filtered image in the original domain.
    '''

    m, n = image_array.shape

    f_transform = fft2(image_array)
    f_shift = fftshift(f_transform)

    y = np.arange(m) - (m // 2)
    x = np.arange(n) - (n // 2)
    xv, yv = np.meshgrid(x, y)

    radius = np.sqrt(xv**2 + yv**2)

    T = ((radius > bandmin) & (radius < bandmax)).astype(float)

    filtered_spectrum = f_shift * T
    filtered_image = np.real(ifft2(ifftshift(filtered_spectrum)))

    return T, filtered_image