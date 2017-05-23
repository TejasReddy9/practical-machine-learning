import pandas as pd
import quandl as Quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

api_key = 'xTyEm3dGzNrDP4w6AVfM'

df = Quandl.get('WIKI/GOOGL',authtoken=api_key)

# print(df.head())
# # stock splits : "10 stocks-1000$ a share", then every share is now doubled then "20 stocks - 500$ each say to buy more".. this is where adjusted values comes up
# # understand relationship among the features .. mostly for Deep Learning. not now

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

# margin of high and low.. .volatility for the day
# open and close price .... by how price went up/ gone down

#we will extract features
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] *100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] *100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# rounding up no. of days_out : for 10 days
forecast_out = int(math.ceil(0.01*len(df)))   
print(forecast_out)   # 32 days ahead forecasting

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

# Test data is 20%, Train data is 80%
X_train, X_test, y_train, y_test = cross_validation.train_test_split( X, y, test_size=0.2)

clf = LinearRegression(n_jobs=10)  # most accurate, can be threaded, n_jobs:no. of threads, 'n_jobs=-1'-- maximum possible threads by our computer
# clf = svm.SVR()   # not so accurate , Support Vector Regression
# clf = svm.SVR(kernel='poly') # default -- linear
clf.fit(X_train, y_train)
# confidence/accuracy
accuracy = clf.score(X_test, y_test)

print(accuracy)

