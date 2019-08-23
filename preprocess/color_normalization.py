#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-16 10:04:08
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $2$

import os
import cv2
import numpy as np


def lab_mean_std(im_input):
    im_lab = cv2.cvtColor(im_input, cv2.COLOR_RGB2LAB)

    mean_lab = np.zeros(3)
    std_lab = np.zeros(3)

    for i in range(3):
        mean_lab[i] = im_lab[:, :, i].mean()
        std_lab[i] = (im_lab[:, :, i] - mean_lab[i]).std()

    return mean_lab, std_lab


def reinhard(im_src, target_mu, target_sigma):
    """
    Performs Reinhard color normalization to transform the color
characteristics of an image to a desired standard.

Parameters:
    im_src : array_like. 
            An RGB image
    target_mu : array_like
            A 3-element array containing the means of the target image channels n LAB color space.
    target_sigma : array_like
            A 3-element array containing the standard deviations of the target image channels in LAB color space.
Returns:
    im_normalized : array_like 
            Color Normalized RGB image
    """
    m = im_src.shape[0]
    n = im_src.shape[1]

    im_lab = cv2.cvtColor(im_src, cv2.COLOR_RGB2LAB)
    # calculate the means of the source image channels in LAB color space
    src_mu = im_lab.sum(axis=0).sum(axis=0) / (m * n)
    # center to zero-mean
    for i in range(3):
        im_lab[:, :, i] = im_lab[:, :, i] - src_mu[i]

    # calculate the standard deviations of the source image channels in LAB color space
    src_sigma = ((im_lab * im_lab).sum(axis=0).sum(axis=0) /
                 (m * n - 1)) ** 0.5
    # scale to unit variance
    for i in range(3):
        im_lab[:, :, i] = im_lab[:, :, i] / src_sigma[i]

    # rescale and recenter to match target statistics
    for i in range(3):
        im_lab[:, :, i] = im_lab[:, :, i] * target_sigma[i] + target_mu[i]

    im_normalized = cv2.cvtColor(im_lab, cv2.COLOR_LAB2RGB)
    return im_normalized


def color_normalization(im_input, im_ref):
    """
    color normalization
    """
    mean_ref, std_ref = lab_mean_std(im_ref)
    print(mean_ref, std_ref)
    im_normalized = reinhard(im_input, mean_ref, std_ref)
    return im_normalized


# def main():
#     path_ref = 'D://data//breast_cancer_HE//patch//1702621_0_0_0.jpg'
#     im_ref = cv2.imread(path_ref)
#     im_ref = cv2.cvtColor(im_ref, cv2.COLOR_BGR2RGB)
#     cv2.imshow('ref', im_ref)

#     path_input = 'D://data//breast_cancer_HE//patch//1702621_54_10240_4096.jpg'
#     im_input = cv2.imread(path_input)
#     im_input - cv2.cvtColor(im_input, cv2.COLOR_BGR2RGB)
#     cv2.imshow('input', im_input)

#     im_norm = color_normalization(im_input, im_ref)
#     cv2.imshow('norm', im_norm)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()
