import numpy as np
import cv2

def threshold(image, mode):
	if mode == 'ostu':
		ret, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	if mode == 'ksw':
		pass

	return ret, th

if __name__ == '__main__':
	path = './/transfer//1702621_376_26624_29696.jpg'
	image = cv2.imread(path, 0)

	ret, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	print(ret)
	cv2.imshow('image', image)
	cv2.imshow('th', th)

	cv2.waitKey (0)
	cv2.destroyAllWindows() 