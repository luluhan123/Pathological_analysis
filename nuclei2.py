from preprocess import ColorDeconvolution, image_enhancement

import cv2
import pywt
import numpy as np

if __name__ == '__main__':
    path = 'D://data//breast_cancer_HE//patch//1702621_0_0_0.jpg'
    # path = 'patch//1702621_0_0_0.jpg'
    image = cv2.imread(path)

    _, He, _ = ColorDeconvolution(path)
    He = (He - He.min()) / (He.max() - He.min()) * 255
    He = He.astype(np.uint8)

    He_enhance = image_enhancement(He)
    coeffs = pywt.dwt2(He_enhance, 'haar')
    cA, (cH, cV, cD) = coeffs
