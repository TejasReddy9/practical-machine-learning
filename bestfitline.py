import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
	# PEMDAS -- Parenthesis Exponent Multiplication Division Addition Subtraction
	m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs*ys)) /
			((np.mean(xs)**2) - np.mean(xs**2)) )
	b = np.mean(ys) - m*np.mean(xs)
	return m, b
	

m, b = best_fit_slope_and_intercept(xs,ys)

print(m, b)

regression_line = [(m*x)+b for x in xs]

# regression_line = []
# for x in xs:
# 	regression_line.append()

predict_x = 8
predict_y = (m*predict_x)+b

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()



