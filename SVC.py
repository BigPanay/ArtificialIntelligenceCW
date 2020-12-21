# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 22:19:20 2020

@author: Chloe
"""

from time import time
import pandas as pd
import numpy as np
#from sklearn import preprocessing
#from sklearn import utils
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#start timer
start = time()
###############

df = pd.read_csv("vgsales2.csv", index_col=0)
col_list = ['Year', 'Global_Sales']
df = df[col_list]
df = df[~(df <= 0.5).any(axis=1)]
print(df.__len__())

XStandard = StandardScaler().fit_transform(df)

kmeans = KMeans(n_clusters=2)
kmeans.fit(XStandard)
print(kmeans.labels_.shape)
print(kmeans.cluster_centers_.shape)
print(XStandard.shape)
plt.figure(figsize=(11,5))

x = XStandard
y = kmeans.labels_


for i, C in enumerate([100, 2000]):
    clf = LinearSVC(C=C, loss="hinge", random_state=42).fit(x, y)
    decision_function = clf.decision_function(x)
    support_vector_indices = np.where((2 * y - 1) * decision_function <= 1)[0]
    support_vectors = x[support_vector_indices]

    plt.subplot(1, 2, i + 1)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')
    plt.title("C=" + str(C))
plt.tight_layout()
plt.show()

###############
end = time()
time_taken = end - start
print("time taken: %.2f" % time_taken, "seconds")