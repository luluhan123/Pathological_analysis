import os
import cv2
import openslide
import numpy as np
from PIL import Image

def split_data(svs_path, patch_path, patch_size, file_index):
	svs_name = svs_path.split('//')[-1]
	svs_name = svs_name.split('\\')[-1]
	name = svs_name.split('.')[0]

	print('start read svs file: ', svs_name)
	
	slide = openslide.open_slide(svs_path)
	downsamples = slide.level_downsamples
	downsamples_num = len(downsamples)
	shapes = slide.level_dimensions

	min_rgba_image = slide.read_region((0,0), downsamples_num - 1, shapes[downsamples_num - 1])
	min_rgba_arr = np.array(min_rgba_image)
	min_rgb_arr = cv2.cvtColor(min_rgba_arr, cv2.COLOR_RGBA2RGB)

	lower_red = np.array([0, 50, 50])
	upper_red = np.array([179, 255, 255])

	patch_lower_red = np.array([0, 50, 20])
	patch_upper_red = np.array([179, 255, 255])

	min_hsv_arr = cv2.cvtColor(min_rgb_arr, cv2.COLOR_RGB2HSV)
	min_bin_arr = cv2.inRange(min_hsv_arr, lower_red, upper_red)

	close_kernel = np.ones((25,25), dtype = np.uint8)
	min_close_arr = cv2.morphologyEx(min_bin_arr, cv2.MORPH_CLOSE, close_kernel)
	open_kernel = np.ones((10, 10), dtype = np.uint8)
	min_open_arr = cv2.morphologyEx(min_close_arr, cv2.MORPH_OPEN, open_kernel)

	contours, _ = cv2.findContours(min_open_arr, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	bounding_boxes = []
	for cnt in contours:
		bounding_boxes.append(cv2.boundingRect(cnt))

	mag_factor = downsamples[downsamples_num - 1]

	for index, bounding_box in enumerate(bounding_boxes):
		b_x_start = int(bounding_box[0]) * mag_factor
		b_y_start = int(bounding_box[1]) * mag_factor
		b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) * mag_factor
		b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) * mag_factor
		b_width = int(bounding_box[2]) * mag_factor 
		b_height = int(bounding_box[3]) * mag_factor

		x_size = int(b_width / patch_size)
		y_size = int(b_height / patch_size)

		for i in range(0,x_size):
			for j in range(0, y_size):
				x = int(b_x_start + i * patch_size)
				y = int(b_y_start + j * patch_size)
				patch_rgba_image = slide.read_region((x,y), 0, (patch_size,patch_size))
				patch_rgba_arr = np.array(patch_rgba_image)
				patch_rgb_arr = cv2.cvtColor(patch_rgba_arr, cv2.COLOR_RGBA2RGB)

				patch_hsv_arr = cv2.cvtColor(patch_rgb_arr, cv2.COLOR_RGB2HSV)
				patch_bin_arr = cv2.inRange(patch_hsv_arr, patch_lower_red, patch_upper_red)
				white_count_in_patch = cv2.countNonZero(patch_bin_arr)

				if (white_count_in_patch / (patch_size * patch_size)) > 0.5:
					Image.fromarray(patch_rgb_arr).save(os.path.join(patch_path, name + '_' + str(file_index) + '_' + str(x) + '_' + str(y) + '.jpg'), 'JPEG')
					print(svs_name, 'saving: ', patch_path + '_' + str(file_index) + '_' + str(x) + '_' + str(y))
					file_index += 1
	print(svs_name, 'over')


if __name__ == '__main__':
	path = './/breast_cancer_HE'
	patch_path = './/patch'
	patch_size = 1024
	svs_list = os.listdir(path)
	file_index = 0

	for svs_name in svs_list:
		name = svs_name.split('.')[0]
		svs_path = os.path.join(path, svs_name)
		print(svs_path)
		split_data(svs_path, patch_path, patch_size, file_index)