import csv
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

nlines=25
ndays=nlines/5*7
stockname='AAPL'
dates = []
prices = []

today=pd.Timestamp.today()

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(row[0])
            prices.append(float(row[1]))
    return

def show_plot(dates, prices):
    linear_mod1 = linear_model.LinearRegression()
    linear_mod2 = linear_model.Ridge(alpha=2000)
    linear_mod3 = linear_model.Lasso(alpha=20)
    dates = np.reshape(dates, (len(dates),1))
    prices = np.reshape(prices, (len(prices),1))
    linear_mod1.fit(dates,prices)
    linear_mod2.fit(dates,prices)
    linear_mod3.fit(dates,prices)
    plt.scatter(dates,prices,color='black')
    plt.plot(dates,linear_mod1.predict(dates), color='red', linewidth=1, label="Linear")
    plt.plot(dates,linear_mod2.predict(dates), color='green', linewidth=1, label="Ridge")
    plt.plot(dates,linear_mod3.predict(dates), color='blue', linewidth=1, label="Lasso")
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel('Day')
    plt.ylabel('Stock Price')
    plt.title('Linear stock prediction '+stockname)
    plt.show()
    
def predict_price(dates, prices, x):
    linear_mod = linear_model.LinearRegression()
    dates = np.reshape(dates, (len(dates),1))
    prices = np.reshape(prices, (len(prices),1))
    linear_mod.fit(dates,prices)
    predicted_price = linear_mod.predict(x)
    return predicted_price[0][0], linear_mod.coef_[0][0], linear_mod.intercept_[0]

get_data(stockname+'.csv')
dates = dates[-nlines:]
prices = prices[-nlines:]
pdates=({})
pdates['daystring'] = dates
pdates['day'] = ndays - (today-pd.to_datetime(dates)).days
#print(dates)
#print(prices)

show_plot(pdates['day'],prices)    
    
predicted_price, coefficient, constant = predict_price(pdates['day'],prices,[[ndays+1]])

print("Linear Regression model")
print("Day",ndays+1)
print("Predicted Price:",predicted_price)
print("ax+b,  a =",coefficient,"  b =",constant)