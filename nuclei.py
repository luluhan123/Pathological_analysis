from preprocess import ColorDeconvolution, sobel_image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

MAX_AREA = 1048576
AREA_THRESHOLD = 8000

if __name__ == '__main__':
    # path = 'D://data//breast_cancer_HE//transfer//1702621_382_27648_5120.jpg'
    path = 'D://data//breast_cancer_HE//patch//1702621_0_0_0.jpg'
    image = cv2.imread(path)

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
    _, th = cv2.threshold(He_closing, 0 ,255,
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

    # find contours
    contours, hierachy = cv2.findContours(
        th_remove_small_objects, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours and show image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    cv2.imshow('test', image)
    cv2.waitKey()
