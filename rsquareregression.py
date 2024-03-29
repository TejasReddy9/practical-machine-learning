# How good is a best fit line -- Coefficient of determination
import numpy as np
import random #psuedorandom
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)

# sample dataset for testing our correctness
# hm - how many datapoints, True--positive correlation, step is negative for negative correlation 
def create_dataset(hm, variance, step=2, correlation=False):
	val = 1;
	ys = []
	for i in range(hm):
		y = val + random.randrange(-variance,variance)
		ys.append(y)
		if correlation and correlation=='pos':
			val += step
		elif correlation and correlation=='neg':
			val -= step
	xs = [i for i in range(len(ys))]

	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs,ys):
	m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs*ys)) /
			((np.mean(xs)**2) - np.mean(xs**2)) )
	b = np.mean(ys) - m*np.mean(xs)
	return m, b
	
def squared_error(ys_orig, ys_line):
	return sum((ys_line - ys_orig)**2)

def coeff_of_determination(ys_orig, ys_line):
	y_mean_line = [np.mean(ys_orig) for y in ys_orig]
	squared_error_regr = squared_error(ys_orig, ys_line)
	squared_error_y_mean = squared_error(ys_orig, y_mean_line)
	return (1 - (squared_error_regr/squared_error_y_mean))


xs, ys = create_dataset(40, 10, 2, correlation='pos') # Check var=10,40,80.. corr='pos','neg',False.. 


m, b = best_fit_slope_and_intercept(xs,ys)

print(m, b)

regression_line = [(m*x)+b for x in xs]

predict_x = 8
predict_y = (m*predict_x)+b

r_squared = coeff_of_determination(ys, regression_line)
print(r_squared)


plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, s=100, color='g')
plt.plot(xs, regression_line)
plt.show()



