import cv2
import numpy as np

# PROBLEM 1: GAUSSIAN FILTERING


def filterGaussian(
    image: np.ndarray,
    kernel_size: int,
    kernel_sigma: float,
    border_type,
    separable: bool,
) -> np.ndarray:
    """**PROBLEM 1: GAUSSIAN FILTERING**

    Args:
        image (numpy.ndarray): input RGB image
        kernel_size (int): an odd integer to specify the kernel size.If this is 5, then the actual kernel size is 5x5.
        kernel_sigma (float): a positive real value to control the shape of the filter kernel.
        border_type (_type_): extrapolation method for handling image boundaries. Possible values are: `cv2.BORDER_CONSTANT`, `cv2.BORDER_REPLICATE`, `cv2.BORDER_REFLECT`, `cv2.BORDER_WRAP`, `cv2.BORDER_REFLECT_101`
        separable (bool): Boolean value. If `separable == true`, then the function performs Gaussian filtering using two 1D filters. Otherwise, the function performs Gaussian filtering using a normal 2D convolution operation.

    Returns:
        image (np.adarray): The function must return a filtered RGB image.
    """

    assert len(image.shape) == 3  # do not accept GrayScale or ARGB image
    assert image.shape[2] == 3  # make sure image has RGB channel
    assert kernel_size % 2 == 1  # kernel size is odd
    assert kernel_sigma > 0  # kernel sigma is positive real number
    assert border_type in [
        cv2.BORDER_CONSTANT,
        cv2.BORDER_REPLICATE,
        cv2.BORDER_REFLECT,
        cv2.BORDER_WRAP,
        cv2.BORDER_REFLECT_101,
    ]

    # height, width from original image, not padded image
    height, width, channel = image.shape
    result = np.zeros_like(image)

    if kernel_size != 1:
        padding_size = kernel_size // 2
        image = cv2.copyMakeBorder(
            image, padding_size, padding_size, padding_size, padding_size, border_type
        )

    if separable:
        kernel = build1DGaussianKernel(kernel_size, kernel_sigma)
        kernel_row = kernel.reshape(1, kernel_size)
        kernel_col = kernel.reshape(kernel_size, 1)
    else:
        kernel = build2DGaussianKernel(kernel_size, kernel_sigma)

    if separable:
        aux = np.zeros(height, width + 2 * padding_size, channel)
        for i in range(height):
            for j in range(width + 2 * padding_size):
                for k in range(channel):
                    aux[i][j][k] = np.sum(image[i : i + kernel_size, j, k] * kernel_col)
        for i in range(height):
            for j in range(width):
                for k in range(channel):
                    result[i][j][k] = np.sum(
                        aux[i, j : j + kernel_size, k] * kernel_row
                    )
    else:
        for i in range(height):
            for j in range(width):
                for k in range(channel):
                    result[i][j][k] = np.sum(
                        image[i : i + kernel_size, j : j + kernel_size, k] * kernel
                    )

    return result


def gaussian1D(sigma: float, a: int) -> float:
    """Gaussian distribution function in 1D. This function does not consider constant term of the formula.

    Args:
        sigma (float): standard deviation of gaussian distribution
        a (int): input of gaussian distribution

    Returns:
        float: desired output of 1D gaussian distribution at given `a` without consideration of constant alpha
    """
    # ignored constant term: 1 / sqrt(2\pi\sigma^2)
    # normalizing will make them useless
    return np.exp(-(a**2) / (2 * sigma**2))


def build1DGaussianKernel(kernel_size: int, kernel_sigma: float) -> np.ndarray:
    """build 1D gaussian kernel with given kernel size and kernel sigman for gaussian distribution

    Args:
        kernel_size (int): _description_
        kernel_sigma (float): _description_

    Returns:
        np.ndarray: (kernel_size, ) 1D gaussian kernel
    """
    mid = kernel_size // 2
    kernel = np.arange(mid + 1, dtype=np.float32)
    kernel = np.vectorize(lambda x: gaussian1D(sigma=kernel_sigma, a=x))(kernel)
    kernel = np.concatenate((kernel[-1:0:-1], kernel))
    return kernel / np.sum(kernel)


def build2DGaussianKernel(kernel_size: int, kernel_sigma: float) -> np.ndarray:
    """build 2D gaussian kernel with given kernel size and kernel sigman for gaussian distribution

    Args:
        kernel_size (int): kernel size n of n by n matrix
        kernel_sigma (float): standard deviation of gaussian distribution

    Returns:
        np.ndarray: (kernel_size, kernel_size) 2D gaussian kernel
    """
    # 2D Gaussian kernel is product of 1D Gaussian kernel and its transpose
    g = build1DGaussianKernel(
        kernel_size=kernel_size, kernel_sigma=kernel_sigma
    ).reshape(kernel_size, 1)
    kernel = np.matmul(g, g.T)
    # suppose g is normalized already, result of g * g.T should be normalized, too
    return kernel


# PROBLEM 2: HISTOGRAM EQUALIZATION


def histogramEqualization(image: np.ndarray) -> np.ndarray:
    """**PROBLEM 2: HISTOGRAM EQUALIZATION**

    Args:
        image (np.ndarray): input image, it should be grayscale or (a)rgb image

    Returns:
        np.ndarray: histogram equalized image, same size as input image
    """
    assert (
        len(image.shape) == 3 or len(image.shape) == 2
    )  # image should be (A)RGB or GRAYSCALE

    if len(image.shape) == 2:  # GRAYSCALE
        return histogramEqulizationSingleChannel(image)

    for c in range(image.shape[2]):
        image[:, :, c] = histogramEqulizationSingleChannel(image[:, :, c])
    return image


def histogramEqulizationSingleChannel(image: np.ndarray) -> np.ndarray:
    """histogram equalization on single channel

    Args:
        image (np.ndarray): input image of (height, width), this should be single channel image

    Returns:
        np.ndarray: histogram equalized image, same size as input image
    """
    assert len(image.shape) == 2

    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    chf = hist.cumsum()  # cumulative histogram function
    coef = 255 / chf[-1]  # transformation coefficient
    return np.vectorize(lambda p: int(coef * chf[p]))(image)
