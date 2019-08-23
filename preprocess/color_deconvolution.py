import numpy as np 
import cv2
import matplotlib.pyplot as plt

def norm(vector):
	n = 0
	for i in vector:
		n = n + np.square(i)
	return np.sqrt(n)

def ColorDeconvolution(path):
	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	[l, c, d] = image.shape
	od = np.zeros([l, c, d])
	for i in range(l):
		for j in range(c):
			for k in range(d):
				if image[i, j, k] != 0:
					od[i, j, k] = np.log(image[i, j, k])

	He = np.array([0.6500286, 0.704031, 0.2860126])  # Hematoxylin
	Eo = np.array([0.07, 0.99, 0.11])  # Eosine
	NA = np.array([0.7565,0.6923,0.9524]) # NA

	HENAtoRGB = np.array([He / norm(He), Eo / norm(Eo), NA / norm(NA)])
	RGBtoHENA = np.linalg.inv(HENAtoRGB)

	stains = np.zeros([l, c, d])
	for i in range(l):
		for j in range(c):
			a = np.dot(od[i, j], RGBtoHENA)
			b = od[i, j]
			stains[i, j, :] = a[:]

	He = stains[:, :, 0]
	Eo = stains[:, :, 1]
	return stains, He, Eo

# if __name__ == '__main__':
# 	path = '..//patch//1702621_0_0_0.jpg'
# 	stains, _, _ = ColorDeconvolution(path)
# 	# print(stains.max(), stains.min(), stains.std())


# 	He = stains[:, :, 0]
# 	plt.subplot(1, 2, 1)
# 	plt.imshow(He, cmap="gray")

# 	Eo = stains[:, :, 1]
# 	plt.subplot(1, 2, 2)
# 	plt.imshow(Eo, cmap="gray")

# 	plt.show()