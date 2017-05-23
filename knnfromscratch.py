from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

#features
dataset = {'k':[[1,2],[2,3],[3,1]],
		   'r': [[6,5],[7,7],[8,6]] }
new_features =[5,7]

for i in dataset:
	for ii in dataset[i]:
		plt.scatter(ii[0],ii[1], s=100, color=i)

# # [[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0],new_features[1],s=100,color='y')
# plt.show()

def k_nearest_neighbors(data, predict, k=3):
	if (len(data) >= k):
		warnings.warn('K is set to a value less than total voting groups!')
	distances = []
	for group in data:
		for features in data[group]:
			# # not so fast
			# euclidean_distance = sqrt( (features[0] - predict[0])**2 + (features[1] - predict[1])**2 )
			# # for more than two groups, but still slow
			# euclidean_distance = np.sqrt(np.sum( (np.array(features) - np.array(predict))**2 ))
			# numpy's simplest version
			euclidean_distance = np.linalg.norm( np.array(features) - np.array(predict) )
			distances.append([euclidean_distance, group])

	votes = []
	for i in sorted(distances)[:k]:
		votes.append(i[1])

	# print(Counter(votes).most_common(1))
	# most_common is tuple of an array.. first one which group is common, second one how many such groups are there
	vote_result = Counter(votes).most_common(1)[0][0]

	return vote_result

result = k_nearest_neighbors(dataset, new_features, k=3)

print(result)

plt.scatter(new_features[0],new_features[1],s=100,color=result)
plt.show()


