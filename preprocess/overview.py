import os
import openslide
import cv2
import numpy as np


if __name__ == '__main__':
	data_path = 'D:\\Han\\Pathological analysis\\breast_cancer_HE'
	overview_path = 'D:\\Han\\Pathological analysis\\overview'

	svsfile_list = os.listdir(data_path)
	for svsfile_name in svsfile_list:
		print('opening file : ', svsfile_name)
		svsfile_path = os.path.join(data_path, svsfile_name)

		slide = openslide.open_slide(svsfile_path)
		downsamples = slide.level_downsamples
		downsamples_num = len(downsamples)
		shapes = slide.level_dimensions

		print('downsamples : ', downsamples)
		print('downsamples number : ', downsamples_num)
		print('shapes : ', shapes)
		print('______________________________________________________')

		min_level_image = slide.read_region((0,0), downsamples_num - 2, shapes[downsamples_num - 2])
		min_level_arr = np.array(min_level_image)
		# cv2.imshow(svsfile_name, min_level_arr)
		cv2.imwrite(os.path.join(overview_path, svsfile_name.split('.')[0]+'X4.jpg'), min_level_arr)

	# cv2.waitKey (0)
	# cv2.destroyAllWindows() 