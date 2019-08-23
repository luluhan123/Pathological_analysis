#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-17 16:43:40
# @Author  : han lulu (han.fire@foxmail.com)
# @Link    : https://github.com/luluhan123/
# @Version : $Id$
# @ref     : Improved Automatic Detection and Segmentation of Cell Nuclei in Histopathology Images

import os
import cv2
import numpy as np


def norm_histogram(image):
    """ compute the normalized image histogram.
    paramters:
            image: array-like 1-D inout image

    returns:
            hist: normalized image histogram
    """
    hist = cv2.calcHist([image], [0], None, [128], [0.0, 255.0])
    return hist
