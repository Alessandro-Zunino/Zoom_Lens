import numpy as np
from scipy import pi
from scipy.signal import fftconvolve

def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)
    
    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    
    return out

def drift(corr):
    N = (corr.shape[0] + 1) / 2
    M = (corr.shape[1] + 1) / 2
    
    idx = np.argwhere(corr==np.max(corr))[0]
    
    nx = np.arange(-N+1, N)
    ny = np.arange(-M+1, M)
    
    dx = nx[idx[0]]
    dy = ny[idx[1]]

    return np.int(dx), np.int(dy)

def driftcorrect(X, drift):
    dx, dy = drift
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0

    return X

def hann2d(shape):
    M, N = shape

    x = np.arange(N)
    y = np.arange(M)
    X, Y = np.meshgrid(x,y)
    W = 0.5 * ( 1 - np.cos( (2*pi*X)/(N-1) ) )
    W *= 0.5 * ( 1 - np.cos( (2*pi*Y)/(M-1) ) )
    return W