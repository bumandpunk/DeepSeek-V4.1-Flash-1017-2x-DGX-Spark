import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

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
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('7.1',4,"/data/test_data.h5")

target=targets[0]
matrix = np.array([[1, 0,0,0], [0,0, 0,1]])
bandmin,bandmax = 20,50
image_array = np.tile(matrix, (400, 200))
assert np.allclose(apply_band_pass_filter(image_array, bandmin,bandmax), target)

target=targets[1]
matrix = np.array([[1, 0,0,0], [0,0, 0,1]])
bandmin,bandmax = 40,80
image_array = np.tile(matrix, (400, 200))
assert np.allclose(apply_band_pass_filter(image_array, bandmin,bandmax), target)

target=targets[2]
matrix = np.array([[1, 0,0,0], [0,0, 0,1]])
bandmin,bandmax = 80,100
image_array = np.tile(matrix, (400, 200))
assert np.allclose(apply_band_pass_filter(image_array, bandmin,bandmax), target)

target=targets[3]
matrix = np.array([[1, 0,0,0], [0,0, 0,1]])
frequency_threshold = 100
bandmin,bandmax = 40,80
T1, filtered_image1 = apply_band_pass_filter(image_array, bandmin,bandmax)
matrix = np.array([[1, 0,0,0], [0,0, 0,1]])
bandmin,bandmax = 60, 80
image_array = np.tile(matrix, (400, 200))
T2, filtered_image = apply_band_pass_filter(image_array, bandmin,bandmax)
MaskAreaDiff = T1.sum()-T2.sum()
assert (MaskAreaDiff > 0) == target
