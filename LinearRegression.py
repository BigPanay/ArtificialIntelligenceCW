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

#path = "../ArtificialIntelligenceCW/" 
#filename_read = os.path.join(path, "vgsales2.csv")
#df = pd.read_csv(filename_read, na_values=['NA', '?'])

df = pd.read_csv (r'C:\Users\Panay\ArtificialIntelligenceCW\vgsales2.csv', na_values=['NA', '?'])

print(df)

print(df.isnull().any())

#collect the columns names for non-target features
result = []
for x in df.columns:
    if x != 'Global_Sales':
        result.append(x)
   
X = df[result].values
y = df['Global_Sales'].values

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)

model = LinearRegression()  
model.fit(X_train, y_train)

print(model.coef_)

y_pred = model.predict(X_test)

df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)

print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('Coefficient of determination: %.2f' % metrics.r2_score(y_test, y_pred))
print('Correlation: ', stats.pearsonr(y_test,y_pred))

plt.rc('figure', figsize=(8, 8))

x = np.linspace(400,0,100)
plt.plot(x, x, '-r',color='r')

plt.scatter(y_test, y_pred, color='black')

plt.xticks(())
plt.yticks(())

plt.show()

def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
    
chart_regression(y_pred,y_test,sort=True)   


def chart_regression2(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
    
chart_regression2(y_pred[:100].flatten(),y_test[:100],sort=True)   

df_head.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

