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
from scicode_helpers import process_hdf5_to_tuple
targets=process_hdf5_to_tuple('6.1',4,"/data/test_data.h5")

target=targets[0]
matrix = np.array([[1, 0], [1, 0]])
frequency_threshold = 20
image_array = np.tile(matrix, (100, 100))
assert np.allclose(apply_low_pass_filter(image_array, frequency_threshold), target)

target=targets[1]
matrix = np.array([[1, 0,0,0], [1,0, 0,0]])
frequency_threshold = 50
image_array = np.tile(matrix, (400, 200))
assert np.allclose(apply_low_pass_filter(image_array, frequency_threshold), target)

target=targets[2]
matrix = np.array([[1, 0,1,0], [1,0, 1,0]])
frequency_threshold = 50
image_array = np.tile(matrix, (400, 200))
assert np.allclose(apply_low_pass_filter(image_array, frequency_threshold), target)

target=targets[3]
matrix = np.array([[1, 0,1,0], [1,0, 1,0]])
frequency_threshold1 = 50
image_array = np.tile(matrix, (400, 200))
image_array1 = np.tile(matrix, (400, 200))
T1, filtered_image1 = apply_low_pass_filter(image_array, frequency_threshold1)
matrix = np.array([[1, 0,0,0], [1,0, 0,0]])
frequency_threshold2 = 40
image_array2 = np.tile(matrix, (400, 200))
T2, filtered_image2 = apply_low_pass_filter(image_array, frequency_threshold2)
MaskAreaDiff = T1.sum()-T2.sum()
assert ((3.1415925*(frequency_threshold1**2-frequency_threshold2**2)-MaskAreaDiff)/MaskAreaDiff<0.01) == target
