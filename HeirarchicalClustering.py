# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:54:57 2020

@author: kingslayer
"""
#Heirarchical Clustering

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv(r"Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

#using dendogram to find optimum no. of clusters
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method="ward"))
plt.title("Dendogram")
plt.ylabel("Euclidian Distane")
plt.xlabel("clusters")
plt.show()

#Fitting hc to dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y_hc=hc.fit_predict(X)

#Visualising the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c="red",label="cluster 1")
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c="green",label="cluster 2")
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c="blue",label="cluster 3")
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c="orange",label="cluster 4")
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c="violet",label="cluster 5")
plt.legend()
plt.title("K-MEANS")
plt.xlabel("Income")
plt.ylabel('Score')
plt.show()
