import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def apply_low_pass_filter(image_array, frequency_threshold):
    m, n = image_array.shape

    f_transform = fft2(image_array)
    f_shifted = fftshift(f_transform)

    center_y, center_x = m // 2, n // 2
    y, x = np.ogrid[:m, :n]
    distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

    T = (distance < frequency_threshold).astype(float)

    filtered_spectrum = f_shifted * T
    filtered_image = np.real(ifft2(ifftshift(filtered_spectrum)))

    return T, filtered_image