import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1,2],
			  [1.5,1.8],
			  [5,8],
			  [8,8],
			  [1,0.6],
			  [9,11]])

# plt.scatter(X[:,0], X[:,1], s=150, linewidth=5)
# plt.show()

colors = ['g','r','c','b','k','y']

class K_Means:
	def __init__(self, k=2, tol=0.001, max_iter=300):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self, data):
		self.centroids = []
		for i in range(self.k):
			self.centroids[i] = data[i]

		for i in range(self.max_iter):

			self.classifications = []
			for i in range(self.k):
				self.classifications[i] = []

			for featureset in self.data:
				distances = []
				for centroid in self.centroids:
					distances.append(np.linalg.norm(featureset - self.centroids[centroid])) # euclidean distance
				classification = distances.index(min(distances))   # minimum distance index will be the index for classification
				self.classifications[classification].append(featureset)

			prev_centroids = self.centroids

			for classification in self.classifications:
				# ........
				self.centroids[classification] = np.average(self.classifications[classification],axis=0)
			# ......

			optimized = True
			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]
				if np.sum((current_centroid - original_centroid)/original_centroid * 100.0) > self.tol:
					optimized = False

			if optimized:
				break


	def predict(self, data):
		distances.append(np.linalg.norm(featureset - self.centroids[centroid])) # euclidean distance
		classification = distances.index(min(distances))
		return classification


clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker='*',color='y',s=70)

for cl


	
















