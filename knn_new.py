# -*- coding: utf-8 -*-
"""
@author: Chloe
"""


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from time import time



#read in the database
df = pd.read_csv("vgsales.csv")
print(df.head())



#start the timer
start = time()


#plot scatter graph
plt.title('Game vs Global Sales, per Year')
salesPlt = df.query("Year < 2020 & Publisher in ['Activision', 'Bethesda Softworks', 'Electronic Arts', 'Microsoft Game Studios', 'Nintendo']").filter(["Year", "Global_Sales", "Publisher"])
sb.scatterplot(salesPlt.iloc[:, 0], salesPlt.iloc[:, 1]/80, hue = salesPlt.iloc[:, 2])



#prepare publisher data
prepPub = df.groupby(["Publisher"]).sum().filter(["EU_Sales"]).sort_values(by = ["EU_Sales"], ascending = False).head(50)
indexPub = {prepPub.index.values[i]:i  for i in range(len(prepPub.index.values))}
data = df.copy()
data["Publisher"].replace(indexPub, inplace = True)
D = [i for i in range(len(prepPub.index.values))]
data = data.drop(data.query("Publisher not in @D").index)
data = data.reset_index(drop = True)



#prepare plaform data
prepPlat = df["Platform"].unique()
indexPlat = {prepPlat[i] : i  for i in range(len(prepPlat))}
data["Platform"].replace(indexPlat, inplace = True)
data.dropna(inplace = True)



#prepare genre data
prepGenre = df["Genre"].unique()
indexGenre = {prepGenre[i] : i  for i in range(len(prepGenre))}
data["Genre"].replace(indexGenre, inplace = True)
data.dropna(inplace = True)


#split data
x = data.reset_index().filter(["Year", "Genre", "Publisher", "EU_Sales", "NA_Sales", "JP_Sales", "Global_Sales", "Other_Sales"]).to_numpy()
y = data.reset_index().filter(["Platform"]).to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)
y_train = y_train.reshape(-1, 1)


#print total element size vs number of testing and training variables
print("{} total elements becomes: {} training variables, and {} testing variables.".format(x.shape[0], x_train.shape[0], x_test.shape[0]))



#print accuracy score
knearest = KNeighborsClassifier(len(prepPub))
knearest.fit(x_train, y_train.ravel())
accuracy = knearest.score(x_test, y_test)
print("accuracy: %.2f" % (accuracy*100), "%")



#end the timer
end = time()
time_taken = end - start
print("time taken: %.2f" % time_taken, "seconds")