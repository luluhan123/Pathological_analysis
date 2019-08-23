#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-18 16:22:40
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$
# @ref     : Improved Automatic Detection and Segmentation of Cell Nuclei in Histopathology Images

from preprocess import ColorDeconvolution, max_clustering, area_open
from preprocess.filters import cdog

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

AREA_THRESHOLD = 8000


def segmentation(path, min_radius=10, max_radius=15, local_max_search_radius=5):
    _, He, _ = ColorDeconvolution(path)
    He = (He - He.min()) / (He.max() - He.min()) * 255
    He = He.astype(np.uint8)

    # open morphological operations
    opening_kernel_size = 5
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))  # 椭圆结构
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size)) # 十字结构
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)) # 矩形结构
    He_opening = cv2.morphologyEx(He, cv2.MORPH_OPEN, opening_kernel)

    # close morphological operations
    closing_kernel_size = 5
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
    He_closing = cv2.morphologyEx(He_opening, cv2.MORPH_CLOSE, closing_kernel)

    # Otsu's thresholding
    _, th = cv2.threshold(He_closing, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.bitwise_not(th)

    # open morphological operations to smooth
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th_opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel1)

    # remove small area and small holes
    th_opening = th_opening.astype(np.bool)
    th_remove_small_objects = morphology.remove_small_objects(th_opening, 80)
    th_remove_small_objects = morphology.remove_small_holes(
        th_remove_small_objects, 100)
    th_remove_small_objects = th_remove_small_objects.astype(np.uint8) * 255

    # find contours
    contours, hierachy = cv2.findContours(
        th_remove_small_objects, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find max area consider as Artifact to fill
    area = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > AREA_THRESHOLD:
            cv2.fillPoly(th_remove_small_objects, [contours[i]], 0)

    # Gradient-weighted distance transform
    # distance transform map
    # dist = cv2.distanceTransform(th_remove_small_objects, cv2.DIST_L2, 3)
    # # gradient map
    # grad = sobel_image(He)
    # grad = np.exp(1 - (grad - grad.min()) / (grad.max() - grad.min()))
    # Dg = np.multiply(dist, grad)

    # run adaptive multi-scale LOG filter
    #vmin_radius = 10
    # max_radius = 15
    im_log_max, im_sigma_max = cdog(
        He, th_remove_small_objects, sigma_min=min_radius * np.sqrt(2), sigma_max=max_radius * np.sqrt(2))

    # detect and segment nuclei using local maximum clustering
    # local_max_search_radius = 5
    im_nuclei_seg_mask, seeds, maxima = max_clustering(
        im_log_max, th_remove_small_objects, local_max_search_radius)

    # filter out small objects
    min_nucleus_area = 80

    im_nuclei_seg_mask = area_open(
        im_nuclei_seg_mask, min_nucleus_area).astype(np.int)
    return im_nuclei_seg_mask

    # print(im_nuclei_seg_mask)
    # plt.imshow(im_nuclei_seg_mask)
    # plt.show()


if __name__ == '__main__':
    # path = 'D://data//breast_cancer_HE//transfer//1702621_382_27648_5120.jpg'
    path = 'D://data//breast_cancer_HE//patch//1702621_0_0_0.jpg'
    # path = 'new.jpg'
    # image = cv2.imread(path)

    for min_radius in range(5, 50, 5):
        for max_radius in range(min_radius, 50, 5):
            for local_max_search_radius in range(3, 20, 1):
                result = segmentation(path, min_radius=min_radius, max_radius=max_radius,
                                      local_max_search_radius=local_max_search_radius)
                plt.imshow(result)
                plt.savefig('result//result' + str(min_radius) + '_' +
                            str(max_radius) + '_' + str(local_max_search_radius) + '.png')
                print('Finished' + str(min_radius) + '_' +
                      str(max_radius) + '_' + str(local_max_search_radius))

    # find contours
    # contours, hierachy = cv2.findContours(
    #     im_nuclei_seg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # draw contours and show image
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    # cv2.imshow('test', image)
    # cv2.waitKey()
