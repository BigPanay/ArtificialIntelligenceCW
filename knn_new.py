# -*- coding: utf-8 -*-
"""

@author: Chloe
"""


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import classification_report

from time import time

#read in the database
df = pd.read_csv("vgsales.csv")
print(df.head())



#start the timer
start = time()


#plot scatter graph
plt.title('Game and its Global Sales per Year')
salesPlt = df.query("Year < 2020 & Publisher in ['Activision', 'Bethesda Softworks', 'Electronic Arts', 'Microsoft Game Studios', 'Nintendo']").filter(["Year", "Global_Sales", "Publisher"])
sb.scatterplot(salesPlt.iloc[:, 0], salesPlt.iloc[:, 1]/80, hue = salesPlt.iloc[:, 2])



#prepare genre data
prepGenre = df.groupby(["Genre"]).sum().filter(["Global_Sales"]).sort_values(by = ["Global_Sales"], ascending = False).head(50)
class_to_genre = {prepGenre.index.values[i]:i  for i in range(len(prepGenre.index.values))}
data = df.copy()
data["Genre"].replace(class_to_genre, inplace = True)
ls = [i for i in range(len(prepGenre.index.values))]
data = data.drop(data.query("Genre not in @ls").index)
data = data.reset_index(drop = True)

#prepare plaform data
prepPlat = df["Platform"].unique()
class_to_plat = {prepPlat[i] : i  for i in range(len(prepPlat))}
data["Platform"].replace(class_to_plat, inplace = True)
data.dropna(inplace = True)

#prepare publisher data
prepPub = df["Publisher"].unique()
class_to_pub = {prepPub[i] : i  for i in range(len(prepPub))}
data["Publisher"].replace(class_to_pub, inplace = True)
data.dropna(inplace = True)



#split data
x = data.reset_index().filter(["Year", "Genre", "Publisher", "EU_Sales", "NA_Sales", "JP_Sales", "Global_Sales", "Other_Sales"]).to_numpy()
y = data.reset_index().filter(["Platform"]).to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)
y_train = y_train.reshape(-1, 1)


#print total element size vs number of testing and training variables
print("{} total elements becomes: {} training variables, and {} testing variables.".format(x.shape[0], x_train.shape[0], x_test.shape[0]))



#print accuracy score
knearest = KNeighborsClassifier(len(prepGenre))
knearest.fit(x_train, y_train.ravel())
accuracy = knearest.score(x_test, y_test)
print("accuracy: %.2f" % (accuracy*100), "%")



#classreport = classification_report(y_test, y_pred)
#print("Report:", classreport)
















#end the timer
end = time()
time_taken = end - start
print("time taken: %.2f" % time_taken, "seconds")