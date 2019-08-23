#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-16 15:13:06
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$

import os
import math

import numpy as np

from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter, gaussian_laplace
from skimage.transform import resize
import numpy as np


def glogkernel(sigma_x, sigma_y, theta):

    N = np.ceil(2 * 3 * sigma_x)
    X, Y = np.meshgrid(np.linspace(0, N, N + 1) - N / 2,
                       np.linspace(0, N, N + 1) - N / 2)
    a = np.cos(theta) ** 2 / (2 * sigma_x ** 2) + \
        np.sin(theta) ** 2 / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + \
        np.sin(2 * theta) / (4 * sigma_y ** 2)
    c = np.sin(theta) ** 2 / (2 * sigma_x ** 2) + \
        np.cos(theta) ** 2 / (2 * sigma_y ** 2)
    D2Gxx = ((2 * a * X + 2 * b * Y)**2 - 2 * a) * \
        np.exp(-(a * X**2 + 2 * b * X * Y + c * Y**2))
    D2Gyy = ((2 * b * X + 2 * c * Y)**2 - 2 * c) * \
        np.exp(-(a * X**2 + 2 * b * X * Y + c * Y**2))
    Gaussian = np.exp(-(a * X**2 + 2 * b * X * Y + c * Y**2))
    Kernel = (D2Gxx + D2Gyy) / np.sum(Gaussian.flatten())
    return Kernel


def cdog(im_input, im_mask, sigma_min, sigma_max, num_octave_levels=3):
    """SCale-adaptive Multiscale Difference-of-Gaussian (DoG) filter for
    nuclei/blob detection.

    Computes the maximal DoG response over a series of scales where in the
    applicable scales at each pixel are constrained to be below an upper-bound
    equal to 2 times the distance to the nearest non-nuclear/background pixel.

    This function uses an approach similar to SIFT interest detection
    where in the scale space between the specified min and max sigma values is
    divided into octaves (scale/sigma is doubled after each octave) and each
    octave is divided into sub-levels. The gaussian images are downsampled by 2
    at the end of each octave to keep the size of convolutional filters small.

    Parameters
    ----------
    im_input : array_like
        A hematoxylin intensity image obtained from ColorDeconvolution. Objects
        are assumed to be dark with a light background.
    mask : array_like
        A binary image where nuclei pixels have non-zero values
    sigma_min : double
        Minumum sigma value for the scale space. For blob detection, set this
        equal to minimum-blob-radius / sqrt(2).
    sigma_max : double
        Maximum sigma value for the scale space. For blob detection, set this
        equal to maximum-blob-radius / sqrt(2).
    num_octave_levels : int
        Number of levels per octave in the scale space.

    Returns
    -------
    im_dog_max : array_like
        An intensity image containing the maximal DoG response accross
        all scales for each pixel
    im_sigma_max : array_like
        An intensity image containing the sigma value corresponding to the
        maximal LoG response at each pixel. The nuclei/blob radius for
        a given sigma value can be calculated as sigma * sqrt(2).

    """
    im_input = im_input.astype(np.float)

    # generate distance map
    im_dmap = distance_transform_edt(im_mask)

    # compute max sigma at each pixel as 2 times the distance to background
    im_sigma_ubound = 2.0 * im_dmap

    # clip max sigma values to specified range
    im_sigma_ubound = np.clip(im_sigma_ubound, sigma_min, sigma_max)

    # compute number of levels in the scale space
    sigma_ratio = 2 ** (1.0 / num_octave_levels)

    k = int(math.log(float(sigma_max) / sigma_min, sigma_ratio)) + 1

    # Compute maximal DoG filter response accross the scale space
    sigma_cur = sigma_min
    im_gauss_cur = gaussian_filter(im_input, sigma_cur)
    im_sigma_ubound_cur = im_sigma_ubound.copy()

    MIN_FLOAT = np.finfo(im_input.dtype).min

    im_dog_max = np.zeros_like(im_input)
    im_dog_max[:, :] = MIN_FLOAT
    im_dog_octave_max = im_dog_max.copy()

    im_sigma_max = np.zeros_like(im_input)
    im_sigma_octave_max = np.zeros_like(im_input)

    n_level = 0
    n_octave = 0

    for i in range(k + 1):

        # calculate sigma at next level
        sigma_next = sigma_cur * sigma_ratio

        # Do cascaded convolution to keep convolutional kernel small
        # G(a) * G(b) = G(sqrt(a^2 + b^2))
        sigma_conv = np.sqrt(sigma_next ** 2 - sigma_cur ** 2)
        sigma_conv /= 2.0 ** n_octave

        im_gauss_next = gaussian_filter(im_gauss_cur, sigma_conv)

        # compute DoG
        im_dog_cur = im_gauss_next - im_gauss_cur

        # constrain response
        im_dog_cur[im_sigma_ubound_cur < sigma_cur] = MIN_FLOAT

        # update maxima
        max_update_pixels = np.where(im_dog_cur > im_dog_octave_max)

        if len(max_update_pixels[0]) > 0:

            im_dog_octave_max[max_update_pixels] = im_dog_cur[max_update_pixels]
            im_sigma_octave_max[max_update_pixels] = sigma_cur

            # print np.unique(im_sigma_octave_max)

        # update cur sigma
        sigma_cur = sigma_next
        im_gauss_cur = im_gauss_next

        # udpate level
        n_level += 1

        # Do additional processing at the end of each octave
        if i == k or n_level == num_octave_levels:

            # update maxima
            if num_octave_levels > 0:

                im_dog_octave_max_rszd = resize(
                    im_dog_octave_max, im_dog_max.shape, order=0)

            else:

                im_dog_octave_max_rszd = im_dog_octave_max

            max_pixels = np.where(
                im_dog_octave_max_rszd > im_dog_max)

            if len(max_pixels[0]) > 0:

                im_dog_max[max_pixels] = \
                    im_dog_octave_max_rszd[max_pixels]

                if num_octave_levels > 0:

                    im_sigma_octave_max_rszd = resize(
                        im_sigma_octave_max, im_dog_max.shape, order=0)

                else:

                    im_sigma_octave_max_rszd = im_sigma_octave_max

                im_sigma_max[max_pixels] = \
                    im_sigma_octave_max_rszd[max_pixels]

            # downsample images by 2 at the end of each octave
            if n_level == num_octave_levels:

                im_gauss_cur = im_gauss_next[::2, ::2]
                im_sigma_ubound_cur = im_sigma_ubound_cur[::2, ::2]

                im_dog_octave_max = im_dog_octave_max[::2, ::2]
                im_sigma_octave_max = im_sigma_octave_max[::2, ::2]

                n_level = 0
                n_octave += 1

    # set min vals to min response
    im_dog_max[im_dog_max == MIN_FLOAT] = 0

    return im_dog_max, im_sigma_max


def clog(im_input, im_mask, sigma_min, sigma_max):
    """Constrainted Laplacian of Gaussian filter.

    Takes as input a grayscale nuclear image and binary mask of cell nuclei,
    and uses the distance transform of the nuclear mask to constrain the LoG
    filter response of the image for nuclear seeding. Returns a LoG filter
    image of type float. Local maxima are used for seeding cells.

    Parameters
    ----------
    im_input : array_like
        A hematoxylin intensity image obtained from ColorDeconvolution. Objects
        are assumed to be dark with a light background.
    im_mask : array_like
        A binary image where nuclei pixels have value 1/True, and non-nuclear
        pixels have value 0/False.
    sigma_min : double
        Minumum sigma value for the scale space. For blob detection, set this
        equal to minimum-blob-radius / sqrt(2).
    sigma_max : double
        Maximum sigma value for the scale space. For blob detection, set this
        equal to maximum-blob-radius / sqrt(2).

    Returns
    -------
    im_log_max : array_like
        An intensity image containing the maximal LoG filter response accross
        all scales for each pixel
    im_sigma_max : array_like
        An intensity image containing the sigma value corresponding to the
        maximal LoG response at each pixel. The nuclei/blob radius value for
        a given sigma can be estimated to be equal to sigma * sqrt(2).

    References
    ----------
    .. [#] Y. Al-Kofahi et al "Improved Automatic Detection and Segmentation
           of Cell Nuclei in Histopathology Images" in IEEE Transactions on
           Biomedical Engineering,vol.57,no.4,pp.847-52, 2010.

    """

    # convert intensity image type to float
    im_input = im_input.astype(np.float)

    # generate distance map
    im_dmap = distance_transform_edt(im_mask)

    # compute max sigma at each pixel as 2 times the distance to background
    im_sigma_ubound = 2.0 * im_dmap

    # clip max sigma values to specified range
    im_sigma_ubound = np.clip(im_sigma_ubound, sigma_min, sigma_max)

    # initialize log filter response array
    MIN_FLOAT = np.finfo(im_input.dtype).min

    im_log_max = np.zeros_like(im_input)
    im_log_max[:, :] = MIN_FLOAT

    im_sigma_max = np.zeros_like(im_input)

    # Compute maximal LoG filter response across the scale space
    sigma_start = np.floor(sigma_min)
    sigma_end = np.ceil(sigma_max)

    sigma_list = np.linspace(sigma_start, sigma_end,
                             sigma_end - sigma_start + 1)

    for sigma in sigma_list:

        # generate normalized filter response
        im_log_cur = sigma**2 * \
            gaussian_laplace(im_input, sigma, mode='mirror')

        # constrain LoG response
        im_log_cur[im_sigma_ubound < sigma] = MIN_FLOAT

        # update maxima
        max_update_pixels = np.where(im_log_cur > im_log_max)

        if len(max_update_pixels[0]) > 0:

            im_log_max[max_update_pixels] = im_log_cur[max_update_pixels]
            im_sigma_max[max_update_pixels] = sigma

    # replace min floats
    im_log_max[im_log_max == MIN_FLOAT] = 0

    return im_log_max, im_sigma_max


def glog(im_input, alpha=1,
         range=np.linspace(1.5, 3, np.round((3 - 1.5) / 0.2) + 1),
         theta=np.pi / 4, tau=0.6, eps=0.6):
    """Performs generalized Laplacian of Gaussian blob detection.

    Parameters
    ----------
    im_input : array_like
        A hematoxylin intensity image obtained from ColorDeconvolution.
    alpha : double
        A positive scalar used to normalize the gLoG filter responses. Controls
        the blob-center detection and eccentricities of detected blobs. Larger
        values emphasize more eccentric blobs. Default value = 1.
    range : array_like
        Scale range
    theta : double
        Angular increment for rotating gLoG filters. Default value = np.pi / 6.
    tau : double
        Tolerance for counting pixels in determining optimal scale SigmaC
    eps : double
        range to define SigmaX surrounding SigmaC

    Returns
    -------
    Rsum : array_like
        Sum of filter responses at specified scales and orientations
    Maxima: : array_like
        A binary mask highlighting maxima pixels

    Notes
    -----
    Return values are returned as a namedtuple

    References
    ----------
    .. [#] H. Kong, H.C. Akakin, S.E. Sarma, "A Generalized Laplacian of
       Gaussian Filter for Blob Detection and Its Applications," in IEEE
       Transactions on Cybernetics, vol.43,no.6,pp.1719-33, 2013.

    """

    # initialize sigma
    Sigma = np.exp(range)

    # generate circular LoG scale-space to determine range of SigmaX
    l_g = 0
    H = []
    Bins = []
    Min = np.zeros((len(Sigma), 1))
    Max = np.zeros((len(Sigma), 1))
    for i, s in enumerate(Sigma):
        Response = s**2 * ndi.filters.gaussian_laplace(im_input, s, output=None,
                                                       mode='constant',
                                                       cval=0.0)
        Min[i] = Response.min()
        Max[i] = Response.max()
        Bins.append(np.arange(0.01 * np.floor(Min[i] / 0.01),
                              0.01 * np.ceil(Max[i] / 0.01) + 0.01, 0.01))
        Hist = np.histogram(Response, Bins[i])
        H.append(Hist[0])
        if Max[i] > l_g:
            l_g = Max[i]

    # re-normalized based on global max and local min, count threshold pixels
    Zeta = np.zeros((len(Sigma), 1))
    for i, s in enumerate(Sigma):
        Bins[i] = (Bins[i] - Min[i]) / (l_g - Min[i])
        Zeta[i] = np.sum(H[i][Bins[i][0:-1] > tau])

    # identify best scale SigmaC based on maximum circular response
    Index = np.argmax(Zeta)

    # define range for SigmaX
    XRange = range(max(Index - 2, 0), min(len(range), Index + 2) + 1)
    SigmaX = np.exp(range[XRange])

    # define rotation angles
    Thetas = np.linspace(0, np.pi - theta, np.round(np.pi / theta))

    # loop over SigmaX, SigmaY and then angle, summing up filter responses
    Rsum = np.zeros(im_input.shape)
    for i, Sx in enumerate(SigmaX):
        YRange = range(0, XRange[i])
        SigmaY = np.exp(range[YRange])
        for Sy in SigmaY:
            for Th in Thetas:
                Kernel = glogkernel(Sx, Sy, Th)
                Kernel *= (1 + np.log(Sx) ** alpha) * (1 + np.log(Sy) ** alpha)
                Rsum += ndi.convolve(im_input, Kernel,
                                     mode='constant', cval=0.0)
                print(Sx, Sy, Th)
        Kernel = glogkernel(Sx, Sx, 0)
        Kernel *= (1 + np.log(Sx) ** alpha) * (1 + np.log(Sx) ** alpha)
        Rsum += ndi.convolve(im_input, Kernel, mode='constant', cval=0.0)
        print(Sx, Sx, 0)

    # detect local maxima
    Disk = morphology.disk(3 * np.exp(range[Index]))
    Maxima = ndi.filters.maximum_filter(Rsum, footprint=Disk)
    Maxima = Rsum == Maxima

    return Rsum, Maxima
