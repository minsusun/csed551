import numpy as np
import cv2

DEFAULT_BORDER_TYPE = cv2.BORDER_REFLECT


def squaredDistanceMatrix(shape: tuple[int, int]) -> "np.ndarray[np.float32]":
    """Get squared distance matrix shaped of given shape

    Args:
        shape (tuple[int, int]): shape of desired distance matrix

    Returns:
        np.ndarray[np.float32]: squared distance matrix
    """
    u: np.ndarray[np.float32] = (
        np.arange(shape[0]).astype(np.float32) - (shape[0] - 1) / 2
    )
    v: np.ndarray[np.float32] = (
        np.arange(shape[1]).astype(np.float32) - (shape[1] - 1) / 2
    )
    U: np.ndarray[np.float32]
    V: np.ndarray[np.float32]
    V, U = np.meshgrid(v, u)  # U, V are both shaped like (u.shape[0], v.shape[0])
    return U**2 + V**2  # distance^2 matrix from the middle


def frequencyDomainFiltering(
    image: "np.ndarray[np.uint8 | np.float32]",
    filter: "np.ndarray[np.uint8 | np.float32]",
) -> "np.ndarray[np.uint8]":
    """Filter given image in frequency domain using FFT

    Args:
        image (np.ndarray[np.uint8 | np.float32]): image to filter
        filter (np.ndarray[np.uint8 | np.float32]): filter to use, filter shape should be same with image in the perspective of width and height

    Returns:
        np.ndarray[np.uint8]: filtered image
    """
    assert len(image.shape) == 3
    assert len(filter.shape) == 2
    assert (
        image.shape[:2] == filter.shape
    )  # image and filter should be matched to be broadcasted

    result_channels: list[np.ndarray[np.uint8]] = []

    for channel in cv2.split(image):
        # move channel to frequency domain
        channel_f: np.ndarray[np.complex128] = np.fft.fft2(channel)
        channel_f_s: np.ndarray[np.complex128] = np.fft.fftshift(channel_f)

        result_f_s: np.ndarray[np.complex128] = channel_f_s * filter

        # move back to spatial domain
        result_f: np.ndarray[np.complex128] = np.fft.ifftshift(result_f_s)
        result: np.ndarray[np.float64] = np.real(np.fft.ifft2(result_f))

        # normalize the values with interval of original channel image
        result: np.ndarray[np.uint8] = cv2.normalize(
            result,
            None,
            np.min(channel),
            np.max(channel),
            cv2.NORM_MINMAX,
            -1,
        ).astype(np.uint8)

        result_channels.append(result)

    return cv2.merge(result_channels)


# PROBLEM 1: IDEAL LOWPASS FILTER


def idealLowPassFilter(
    shape: tuple[int, int],
    threshold: float,
) -> "np.ndarray[np.float32]":
    """Build `shape` Shaped **Ideal Low Pass Filter** with Threshold.

    Ideal Low Pass Filter can be formulated below:

    f(u,v) = 1 if D(u,v) <= D_0

    f(u,v) = 0 otherwise

    Args:
        shape (tuple[int, int]): desired shape of ideal low pass filter
        threshold (float): threshold for ideal low pass filter, D_0

    Returns:
        np.ndarray[np.float32]: Ideal Low Pass Filter
    """
    D: np.ndarray[np.float32] = squaredDistanceMatrix(shape)
    return (D <= threshold**2).astype(np.float32)


def idealLowPassFiltering(
    image: "np.ndarray[np.uint8]",
    padding: int,
    threshold: float,
) -> "np.ndarray[np.uint8]":
    """Filter the image with ideal low pass filter

    Args:
        image (np.ndarray[np.uint8]): image to filter
        padding (int): size of padding
        threshold (float): threshold of filter, D_0

    Returns:
        np.ndarray[np.uint8]: filtered image
    """
    assert len(image.shape) == 3  # Colored Image
    assert image.shape[2] == 3  # RGB
    assert padding >= 0
    assert threshold >= 0

    padded_image: np.ndarray[np.uint8] = cv2.copyMakeBorder(
        image,
        padding,
        padding,
        padding,
        padding,
        DEFAULT_BORDER_TYPE,
    )

    # ideal low pass filter
    ilf: np.ndarray[np.float32] = idealLowPassFilter(padded_image.shape[:2], threshold)

    # result is (height+2*padding, width+2*padding, channels) shaped
    result: np.ndarray[np.uint8] = frequencyDomainFiltering(padded_image, ilf)

    return result[padding:-padding, padding:-padding, :]


# PROBLEM 2: GAUSSIAN LOWPASS FILTER


def gaussianLowPassFilter(
    shape: tuple[int, int],
    threshold: float,
) -> "np.ndarray[np.float32]":
    """Build `shape` Shaped **Gaussian Low Pass Filter** with Threshold.

    Gaussian Low Pass Filter can be formulated below:

    f(u,v) = exp(-D(u,v)^2/D_0^2)

    Args:
        shape (tuple[int, int]): desired shape of gaussian low pass filter
        threshold (float): threshold for gaussian low pass filter, D_0

    Returns:
        np.ndarray[np.float32]: Gaussian Low Pass Filter
    """
    D: np.ndarray[np.float32] = squaredDistanceMatrix(shape)
    return np.exp(-D / threshold**2)


def gaussianLowPassFiltering(
    image: "np.ndarray[np.uint8]",
    padding: int,
    threshold: float,
) -> "np.ndarray[np.uint8]":
    """Filter the image with gaussian low pass filter

    Args:
        image (np.ndarray[np.uint8]): image to filter
        padding (int): size of padding
        threshold (float): threshold of filter, D_0

    Returns:
        np.ndarray[np.uint8]: filtered image
    """
    assert len(image.shape) == 3  # Colored Image
    assert image.shape[2] == 3  # RGB
    assert padding >= 0
    assert threshold >= 0

    padded_image: np.ndarray[np.uint8] = cv2.copyMakeBorder(
        image,
        padding,
        padding,
        padding,
        padding,
        DEFAULT_BORDER_TYPE,
    )

    # gaussian low pass filter
    glf: np.ndarray[np.float32] = gaussianLowPassFilter(
        padded_image.shape[:2], threshold
    )

    # result is (height+2*padding, width+2*padding, channels) shaped
    result: np.ndarray[np.uint8] = frequencyDomainFiltering(padded_image, glf)

    return result[padding:-padding, padding:-padding, :]


# PROBLEM 3: UNSHARP MASKING & CONVOLUTION THEOREM


def psf2otf(
    filter: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Pad and shift the filter, then return with result of FFT of it

    Args:
        filter (np.ndarray): psf, filter
        shape (tuple[int, int]): desired shape of output

    Returns:
        np.ndarray: 2d numpy array otf
    """

    top = filter.shape[0] // 2
    bottom = filter.shape[0] - top
    left = filter.shape[1] // 2
    right = filter.shape[1] - left

    psf = np.zeros(shape, dtype=filter.dtype)

    psf[:bottom, :right] = filter[top:, left:]
    psf[:bottom, shape[1] - left :] = filter[top:, :left]
    psf[shape[0] - top :, :right] = filter[:top, left:]
    psf[shape[0] - top :, shape[1] - left :] = filter[:top, :left]

    # return otf
    return np.fft.fft2(psf)


def unsharpMasking(
    image: np.ndarray,
    domain: str,
) -> np.ndarray:
    assert len(image.shape) == 3  # Colored Image
    assert image.shape[2] == 3  # RGB
    assert domain in ["spatial", "frequency"]

    pass
