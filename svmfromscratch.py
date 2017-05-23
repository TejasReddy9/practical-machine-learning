import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')

class Support_Vector_Machine:
	def __init__(self, visualization=True):
		self.visualization = visualization
		self.colors = {1:'r',-1:'b'}
		if(self.visualization):
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(1,1,1)


	#train
	def fit(self, data):
		self.data = data
		# { ||w|| : [w, b] }
		opt_dict = {}
		transforms = [[1,1],
					  [-1,1],
					  [-1,-1],
					  [1,-1]]	  
		# maximum and minimum ranges
		all_data = []
		for group in self.data:
			for featureset in self.data[group]:
				for feature in featureset:
					all_data.append(feature)
		self.max_feature_value = max(all_data)
		self.min_feature_value = min(all_data)
		all_data = None # get rid of consumed data

		# step till support vectors 'yi(xi.w+b)' nearly equal to 1..
		# are these threadable... No
		step_sizes = [self.max_feature_value*0.1, 
					  self.max_feature_value*0.01, 
					  # expensive
					  self.max_feature_value*0.001
					  ]
		# extremely expensive.. faster for b_range_multiple=2
		# b_range_multiple = 5
		b_range_multiple = 2
		# for b, we dont need to take step size as small as that of w 
		b_multiple = 5

		latest_optimum = self.max_feature_value*10  # hard_coded 10......

		# this can be threaded
		for step in step_sizes:
			w = np.array([latest_optimum,latest_optimum])
			optimized = False  # as this is convex programming.. only one global minimum
			while not optimized:
				# np.arange() is similar to range() # this can be threaded, thumbsup
				for b in np.arange(-1*(self.max_feature_value*b_range_multiple), self.max_feature_value*b_range_multiple, step*b_multiple ):
					for transformation in transforms:
						w_t = w*transformation
						feasibility = True
						# weakest link in SVM, SMO can fix this a bit
						# yi(xi.w + b) >= 1 constraint
						# add break here later... 
						for i in self.data:
							for xi in self.data[i]:
								yi=i
								# for all values, this constriant must be satisfied, otherwise don't accept
								if not np.all(yi*(np.dot(w_t,xi)+b) >= 1):  
									feasibility =False
								# print(xi,':', yi*(np.dot(w_t,xi)+b) >= 1)					
						if feasibility:
							# { ||w|| : [w, b] }
							opt_dict[np.linalg.norm(w_t)] = [w_t,b]
				if w[0]<0:
					optimized = True
					print('Optimized a step')
				else:
					# w = [5,5]  step=1   w-step=[4,4]
					w = w - step
			norms = sorted([n for n in opt_dict])
			opt_choice = opt_dict[norms[0]]
			self.w = opt_choice[0]
			self.b = opt_choice[1]
			latest_optimum = opt_choice[0][0] + step*2
		# just printing the values and checking how close to 1 are they
		for i in self.data:
			for xi in self.data[i]:
				yi=i
				print(xi, yi*(np.dot(self.w,xi)+self.b) )					


	#test
	def predict(self, features):
		# sign of (x.w + b)
		classification = np.sign(np.dot(np.array(features),self.w) + self.b)
		if classification != 0 and self.visualization:
			self.ax.scatter(features[0],features[1],s=200,marker='*',c=self.colors[classification])
		return classification


	def visualize(self):
		for i in data_dict:
			for x in data_dict[i]:
				self.ax.scatter(x[0],x[1],s=100,c=self.colors[i]) 
		# hyperplane --- v = x.w + b 
		# positive support vectors -- v=1
		# neg. support vectors -- v=-1
		# decision boundary -- v=0
		def hyperplane(x,w,b,v):
			return (-w[0]*x-b+v)/w[1]
		datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
		hyp_x_min = datarange[0]
		hyp_x_max = datarange[1]

		# (w.x + b) = 1
		psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
		psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
		self.ax.plot([hyp_x_min,hyp_x_max], [psv1,psv2], 'k')
		# (w.x + b) = -1
		nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
		nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
		self.ax.plot([hyp_x_min,hyp_x_max], [nsv1,nsv2], 'k')
		# (w.x + b) = 0
		db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
		db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
		self.ax.plot([hyp_x_min,hyp_x_max], [db1,db2], 'y--', linewidth=2.5) # yellow dashes

		plt.show()


data_dict = {-1:np.array([[1,7],
						  [2,8],
						  [3,8]]),
			  1:np.array([[5,1],
						  [6,-1],
						  [7,3]])}


svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us =   [[0,10],
				[1,3],
				[3,4],
				[3,5],
				[5,5],
				[5,6],
				[6,-5],
				[5,8]]
for p in predict_us:
	svm.predict(p)

svm.visualize()







