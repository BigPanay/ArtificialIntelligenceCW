# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:10:33 2020

@author: Panay
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVR

#load the video game sales dataset   

path = "../ArtificialIntelligenceCW/" 
filename_read = os.path.join(path, "vgsalesencode.csv")
gameSales = pd.read_csv(filename_read, na_values=['NA', '?'])

#output the dataset and find any missing fields

print(gameSales)   
print(gameSales.isnull().any())

#defining what goes in what axis
result = []
for x in gameSales.columns:
    if x != 'Global_Sales':
        result.append(x)
   
X = gameSales[result].values
y = gameSales['Global_Sales'].values

print(X.shape)

#splitting the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9)

#making a linear model and fitting it
linRegModel = LinearRegression()  
linRegModel.fit(X_train, y_train)

print(linRegModel.coef_)

#predicting global sales with the test data
y_pred = linRegModel.predict(X_test)

#compare real and predicted data
gameSales_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
gameSales_head = gameSales_compare.head(25)
print(gameSales_head)

#output stats
print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('Coefficient of determination: %.2f' % metrics.r2_score(y_test, y_pred))
print('Correlation: ', stats.pearsonr(y_test,y_pred))


#charts and graphs 
plt.rc('figure', figsize=(6, 6))

x = np.linspace(40,0,40)
plt.plot(x, x, '-r',color='r')

plt.scatter(y_test, y_pred, color='black')

plt.show()

def chartRegression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
    
chartRegression(y_pred,y_test,sort=True)   

chartRegression(y_pred[:100].flatten(),y_test[:100],sort=True)   

gameSales_head.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.scatter(y_test, y_pred)

#using a svm to see if it improves the predictions
svrGamesales = SVR(kernel='linear',C=1000, epsilon=1).fit(X_train, y_train)

y_svr_pred = svrGamesales.predict(X_test)

print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_svr_pred)))

chartRegression(y_svr_pred,y_test,sort=True)  