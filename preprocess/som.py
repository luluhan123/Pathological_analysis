import numpy as np
from sklearn.decomposition import PCA

class SOM():
	def __init__(self, x, y, n_class):
		self.map = []
		self.n_neurons = x*y
		self.sigma = x
		self.template = np.arange(x*y).reshape(self.n_neurons,1)
		self.alpha = 0.6
		self.alpha_final = 0.1
		self.shape = [x,y]
		self.epoch = 0
		self.n_class = n_class

	def train(self, X, iter, batch_size=1):
		if len(self.map) == 0:
			x,y = self.shape
			# first init the map
			self.map = np.zeros((self.n_neurons, len(X[0])))

			# then pricipal components of the input data
			eigen = PCA(self.n_class).fit_transform(X.T).T

			# then set different point on the map equal to principal components to force diversification
			self.map[0] = eigen[0]
			self.map[y-1] = eigen[1]
			self.map[(x-1)*y] = eigen[2]
			self.map[x*y - 1] = eigen[3]
			for i in range(4, self.n_class):
				self.map[np.random.randint(1, self.n_neurons)] = eigen[i]

		self.total = iter

		# coefficient of decay for learning rate alpha
		self.alpha_decay = (self.alpha_final/self.alpha)**(1.0/self.total)

		# coefficient of decay for gaussian smoothing
		self.sigma_decay = (np.sqrt(self.shape[0])/(4*self.sigma))**(1.0/self.self.total)

		samples = np.arange(len(x))
		np.random.shuffle(samples)

		for i in range(0, iter):
			idx = samples[i:i+batch_size]
			self.iterate(X[idx])

	def transform(self, X):
		# compute the dot product of the input with the transpose of the map to get the new input vectors
		res = np.dot(np.exp(X), np.exp(self.map.T))/np.sum(np.exp(self.map), axis=1)
		res = res / (np.exp(np.max(res)) + 1e-8)
		return res

	def iterate(self, vector):
		x, y = self.shape

		delta = self.map - vector

		# Euclidan distance of each neurons woth the example
		dists = np.sum((delta)**2, axis=1).reshape(x,y)

		# Best maching unit
		idx = np.argmin(dists)
		print("Epoch ", self.epoch, ": ", (idx/x, idx%y), "; Sigma: ", self.sigma, "; alpha: ", self.alpha)

		# Linearly reducing the width of Gaussian Kernel
		self.sigma = self.sigma*self.sigma_decay
		dist_map = self.template.reshape(x,y)

		# Distance of each neurons in the map from the best matching neuron
		dists = np.sqrt((dist_map/x - idx/x)**2 + (np.mod(dist_map,x) - idx%y)**2).reshape(self.n_neurons, 1)

		# Applying Gaussian smoothing to distances of neurons from best matching neuron
		h = np.exp(-(dists/self.sigma)**2)

		# Updating neurons in the map
		self.map -= self.alpha*h*delta

		# Decreasing alpha
		self.alpha = self.alpha*self.alpha_decay

		self.epoch = self.epoch + 1

