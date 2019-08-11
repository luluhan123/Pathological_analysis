import cv2
import os

if __name__ == '__main__':
	path = './/overview'
	gray_path = './/gray'
	file_list = os.listdir(path)

	for file_name in file_list:
		file_path = os.path.join(path, file_name)
		image = cv2.imread(file_path)
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		cv2.imwrite(os.path.join(gray_path, file_name.split('.')[0]+'_gray.jpg'), image_gray)