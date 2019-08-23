import numpy as np
import cv2
import os

def hsv_norm(path, new_path):
	name = path.split('\\')[-1]
	new_path = os.path.join(new_path, name)

	image = cv2.imread(path)
	hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	img_h = hsv_image[..., 0]
	img_s = hsv_image[..., 1]
	img_v = hsv_image[..., 2]

	img_h_max = img_h.max()
	img_h_min = img_h.min()
	img_s_max = img_s.max()
	img_s_min = img_s.min()
	img_v_max = img_v.max()
	img_v_min = img_v.min()

	img_h = (img_h - img_h_min) / (img_h_max - img_h_min) * 179
	img_s = (img_s - img_s_min) / (img_s_max - img_s_min) * 255
	img_v = (img_v - img_v_min) / (img_v_max - img_s_min) * 255
	hsv_image[..., 0] = img_h
	hsv_image[..., 1] = img_s
	hsv_image[..., 2] = img_v

	image_new = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
	cv2.imwrite(new_path, image_new)

if __name__ == '__main__':
	path = 'D:\\data\\PAIP_Temp\\normal_new_patch'
	file_list = os.listdir(path)
	i = 0
	number = len(file_list)
	new_path = 'D:\\data\\PAIP_Temp\\norm_normal_new_patch'
	for file_name in file_list:
		old_path = os.path.join(path, file_name)
		hsv_norm(old_path, new_path)
		i = i + 1
		print(file_name, ' over', i, '/', number)