#!/usr/bin/python3
import numpy as np
from scipy.ndimage import convolve, correlate
from scipy.fftpack import fft2, ifft2
from typing import Tuple, Dict
import cv2

def conjgrad(x: np.ndarray, b: np.ndarray, max_iter: int, tol: float, A: callable, params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Conjugate gradient solver.

    Parameters:
    x (np.ndarray): Initial guess vector.
    b (np.ndarray): Right-hand side vector.
    max_iter (int): Maximum number of iterations.
    tol (float): Tolerance for convergence.
    A (function): Function to compute the matrix-vector product.
    params (Dict[str, np.ndarray]): Parameters for the A function.

    Returns:
    np.ndarray: Solution vector.
    """
    r = b - A(x, params)
    p = r.copy()
    rsold = np.dot(r, r)

    for _ in range(max_iter):
        Ap = A(p, params)
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    
    return x

def fftconv2(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Perform convolution in the frequency domain.

    Parameters:
    img (np.ndarray): Input image.
    psf (np.ndarray): Point spread function.

    Returns:
    np.ndarray: Result of the convolution.
    """
    return np.real(ifft2(fft2(img) * psf2otf(psf, img.shape)))

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert PSF to OTF (Optical Transfer Function).

    Parameters:
    psf (np.ndarray): Point spread function.
    shape (Tuple[int, int]): Desired shape of the OTF.

    Returs:
    np.ndarray: Optical transfer function.
    """
    psf_top_half = psf.shape[0]//2
    psf_bottom_half = psf.shape[0] - psf_top_half
    psf_left_half = psf.shape[1]//2
    psf_right_half = psf.shape[1] - psf_left_half
    psf_padded = np.zeros(shape, dtype=psf.dtype)
    psf_padded[:psf_bottom_half, :psf_right_half] = psf[psf_top_half:, psf_left_half:]
    psf_padded[:psf_bottom_half, shape[1]-psf_left_half:] = psf[psf_top_half:, :psf_left_half]
    psf_padded[shape[0]-psf_top_half:, :psf_right_half] = psf[:psf_top_half, psf_left_half:]
    psf_padded[shape[0]-psf_top_half:, shape[1]-psf_left_half:] = psf[:psf_top_half, :psf_left_half]
    return fft2(psf_padded)


def deconv_L2(blurred: np.ndarray, latent0: np.ndarray, psf: np.ndarray, reg_strength: float, weight_x: np.ndarray, weight_y: np.ndarray) -> np.ndarray:
    """
    Perform non-blind deconvolution using an L2 norm on the image gradient.

    Parameters:
    blurred (np.ndarray): The blurred image.
    latent0 (np.ndarray): Initial solution for the latent image.
    psf (np.ndarray): Blur kernel (point spread function).
    reg_strength (float): Regularization strength.
    weight_x (np.ndarray): Weight map for the x-direction gradient.
    weight_y (np.ndarray): Weight map for the y-direction gradient.

    Returns:
    np.ndarray: The deblurred image.
    """
    img_size = blurred.shape

    # Image gradient filters
    dxf = np.array([[0, -1, 1]])
    dyf = np.array([[0], [-1], [1]])

    latent = latent0.copy()

    # Compute the conjugate of the PSF by flipping it upside-down and left-and-right
    psf_conj = np.flip(psf)

    # compute b by convoling the blurred image with the conjugate PSF
    b = fftconv2(blurred, psf_conj)
    b = b.ravel()

    # set x as the raveled latent image
    x = latent.ravel()

    # Parameters for the conjugate gradient method
    cg_param: Dict[str, np.ndarray] = {
        'psf': psf,
        'psf_conj': psf_conj,
        'img_size': img_size,
        'reg_strength': reg_strength,
        'weight_x': weight_x,
        'weight_y': weight_y,
        'dxf': dxf,
        'dyf': dyf
    }

    # Run the conjugate gradient method
    x = conjgrad(x, b, 25, 1e-4, Ax, cg_param)

    # Reshape the solution back to the original image size
    latent = x.reshape(img_size)
    return latent

def Ax(x: np.ndarray, p: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Matrix-vector multiplication used in the conjugate gradient method.

    Parameters:
    x (np.ndarray): Input vector.
    p (Dict[str, np.ndarray]): Parameters containing PSF, weights, and gradients.

    Returns:
    np.ndarray: Result of the multiplication.
    """
    x = x.reshape(p['img_size'])
    # Convolve with PSF and its conjugate
    y = fftconv2(fftconv2(x, p['psf']), p['psf_conj'])
    # Add regularization terms
    y += p['reg_strength'] * convolve(p['weight_x'] * correlate(x, p['dxf'], mode='wrap'), p['dxf'], mode='wrap')
    y += p['reg_strength'] * convolve(p['weight_y'] * correlate(x, p['dyf'], mode='wrap'), p['dyf'], mode='wrap')
    return y.ravel()


def demo():
    # demo code for deconvolution
	# this demo code shows deconvolution of a grayscale image
	# you need to implement deconvolution code for RGB images!

    # load an image
    img = cv2.imread('summerhouse.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)/255

    # load a PSF (blur kernel)
    psf = cv2.imread('psf.png')
    psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
    psf = psf.astype(np.float32)
    psf /= np.sum(psf)

	# weight_x and weight_y correspond to the diagonal matrices W_x and W_y in Homework5.pdf
	# Below, both weight_x and weight_y are set to matrices whose all elements are 1.
	# This is equivalent to setting W_x and W_y to diagonal matrices whose diagonal elements are 1.
	# The parameter 'reg_strength' of deconv_L2, which is set to 0.0001 below, corresponds to 2*alpha in Homework5.pdf
    weight_x = np.ones(img.shape)
    weight_y = np.ones(img.shape)

    # perform deconvolution
    deblurred = deconv_L2(img, img, psf, 0.0001, weight_x, weight_y)

    # store the deblurring result
    cv2.imwrite('deblurred.png', deblurred*255)



if __name__ == '__main__':
    demo()

