#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-11 17:13:34
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $1$

import cv2
import numpy as np


def top_bottom_hat_transfer(img):
    '''
    param:
            img: gray image
    '''
    opening_kernel_size = 5
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))  # 椭圆结构

    img_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, opening_kernel)
    img_bottomhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, opening_kernel)
    TB_hat = img + img_tophat - img_bottomhat
    return TB_hat


def image_enhancement(img):
    '''
    param:
            img: gray image
    '''
    TB_hat = top_bottom_hat_transfer(img)
    return TB_hat


# def main():
#     path = 'D://data//breast_cancer_HE//patch//1702621_0_0_0.jpg'
#     image = cv2.imread(path, 0)
#     result = image_enhancement(image)
#     cv2.imshow('result', result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()
