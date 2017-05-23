import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1,2],
			  [1.5,1.8],
			  [5,8],
			  [8,8],
			  [1,0.6],
			  [9,11]])
# # X[:,0] means all zeroeth elements in X array,
# # X[:,1] means all first elements in X array
# plt.scatter(X[:,0], X[:,1], s=150, linewidth=5)
# plt.show()

clf = KMeans(n_clusters=2) # try 6,8
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_   # y

# color and marker together
colors = ['g.','r.','c.','b.','k.','y.']

for i in range(len(X)):
	plt.plot(X[i][0],X[i][1], colors[labels[i]], markersize=10)  # colors-- 0 or 1 (g or r)

plt.scatter(centroids[:,0],centroids[:,1], marker='x',s=50)
plt.show()

