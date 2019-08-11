import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
	filepath = './/transfer//1702621_0_0_0.jpg'
	image = cv2.imread(filepath)
	shape = image.shape

	# 展平
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image_flat = image.reshape((-1,3))
	image_flat = np.float32(image_flat)

	# 迭代参数
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
	flags = cv2.KMEANS_RANDOM_CENTERS

	# 聚类
	K=4
	compactness, labels, centers = cv2.kmeans(image_flat, K, None, criteria, 10, flags)

	# 显示结果
	image_output = labels.reshape((shape[0], shape[1]))
	plt.subplot(121), plt.imshow(image), plt.title('input')
	plt.subplot(122), plt.imshow(image_output), plt.title('kmeans')
	plt.show()