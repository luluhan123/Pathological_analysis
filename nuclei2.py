from preprocess import ColorDeconvolution

import cv2
import numpy as np 

if __name__ == '__main__':
	path = 'transfer//1702621_382_27648_5120.jpg'
	# path = 'patch//1702621_0_0_0.jpg'
	image = cv2.imread(path)

	_, He, _ = ColorDeconvolution(path)
	He = (He - He.min()) / (He.max() - He.min()) * 255
	He = He.astype(np.uint8)