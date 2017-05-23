import pandas as pd
import quandl as Quandl
import math, time
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


api_key = 'xTyEm3dGzNrDP4w6AVfM'

df = Quandl.get('WIKI/GOOGL',authtoken=api_key)
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] *100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] *100.0

# direct dependence of price
#          price         x          x            x   
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))    #considering holidays... mistake 
# forecast_out = int(math.ceil(0.01*len(df)))    # trying to forecast 1% of next data set of this much size in due time
#print(forecast_out)  

df['label'] = df[forecast_col].shift(-forecast_out)
# print(df.head())
# print(df.tail())
# print(df.drop(['label'],1))

X = np.array(df.drop(['label','Adj. Close'],1)) # to illustrate the price dependency
# X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]   # data on which no test or train going to be done
X = X[:-forecast_out]
df.dropna(inplace=True)
# print(df.head())
# print(df.tail())
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split( X, y, test_size=0.2)

clf = LinearRegression(n_jobs=10)  
clf.fit(X_train, y_train)
# confidence/accuracy
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

# my test
# # print(df.iloc[0])  # returns Series of first row
# # print(df.iloc[0].name)   # return index at integer location 0

#hard coding
last_date = df.iloc[-1].name
# last_unix = last_date.Timestamp()
last_unix = time.mktime(dt.datetime.strptime(str(last_date), "%Y-%m-%d %H:%M:%S").timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = dt.datetime.fromtimestamp(next_unix)
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
	next_unix += one_day


# print(df.head())
# print(df.tail())
df['Adj. Close'].plot()
df['Forecast'].plot()

plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()







