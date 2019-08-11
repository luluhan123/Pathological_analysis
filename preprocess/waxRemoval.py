import os
import cv2
import openslide
import numpy as np
from PIL import Image

def waxRemoval(path):
	svs_list = os.listdir(path)
	bounding_path = './/bounding_breast_cancer_HE'
	for svs_name in svs_list:
		svs_path = os.path.join(path, svs_name)

		slide = openslide.open_slide(svs_path)
		downsamples = slide.level_downsamples
		downsamples_num = len(downsamples)
		shapes = slide.level_dimensions

		min_rgba_image = slide.read_region((0,0), downsamples_num - 1, shapes[downsamples_num - 1])
		min_rgba_arr = np.array(min_rgba_image)
		min_rgb_arr = cv2.cvtColor(min_rgba_arr, cv2.COLOR_RGBA2RGB)

		lower_red = np.array([0, 50, 50])
		upper_red = np.array([179, 255, 255])

		min_hsv_arr = cv2.cvtColor(min_rgb_arr, cv2.COLOR_RGB2HSV)
		min_bin_arr = cv2.inRange(min_hsv_arr, lower_red, upper_red)

		close_kernel = np.ones((25,25), dtype = np.uint8)
		min_close_arr = cv2.morphologyEx(min_bin_arr, cv2.MORPH_CLOSE, close_kernel)

		open_kernel = np.ones((10, 10), dtype = np.uint8)
		min_open_arr = cv2.morphologyEx(min_close_arr, cv2.MORPH_OPEN, open_kernel)

		contours, _ = cv2.findContours(min_open_arr, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		min_rgb_arr_temp = min_rgb_arr
		cv2.drawContours(min_rgb_arr_temp,contours,-1,(0,255,0),2)

		bounding_boxes = []
		for cnt in contours:
			bounding_boxes.append(cv2.boundingRect(cnt))

		for bounding in bounding_boxes:
			cv2.rectangle(min_rgb_arr_temp,(int(bounding[0]),int(bounding[1])),(int(bounding[0])+int(bounding[2]),int(bounding[1])+int(bounding[3])),(0,0,255),2)
		svs_file_name = svs_name.split('.')[0]
		Image.fromarray(min_rgb_arr_temp).save(os.path.join(bounding_path, svs_file_name+'.jpg'), 'JPEG')

path = './/breast_cancer_HE'
waxRemoval(path)