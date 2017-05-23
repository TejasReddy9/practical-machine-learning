## clustering --- k nearest neighbors

# Breast Cancer example from UCI-Machine Learning Datasets 
# Class : (2 for benign, 4 for malignant)

import numpy as np
# cross_validation is moved to model_selection
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd


accuracies = []

for i in range(25):
	df = pd.read_csv('breast-cancer-wisconsin.data.txt')
	df.replace('?', -99999,inplace=True) # outlier, dropna can be used
	df.drop(['id'],1,inplace=True) # id has nothing to do wth our prediction

	X = np.array(df.drop(['class'],1))
	y = np.array(df['class'])

	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

	clf = neighbors.KNeighborsClassifier(n_jobs=1)   #can be threaded, default=1, n_jobs=-1 for maximum possible multi-threading 
	clf.fit(X_train, y_train)

	accuracy = clf.score(X_test, y_test)
	# print(accuracy)

	example_measures = np.array([4,2,1,1,1,2,3,2,1])
	example_measures = example_measures.reshape(1,-1)   
	# example_measures = example_measures.reshape(len(example_measures),-1)   
	# Deprication error :  Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
	# (2,-1) -- (len(example_measures),-1) .. two patients predict 

	prediction = clf.predict(example_measures)
	# print(prediction)
	accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))





