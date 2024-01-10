#!/usr/bin/env python3
"""
Created on Sun Dec 17 15:51:45 2023

@author: lijinju
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd


X = np.array([[1, 2],
              [1.5, 2],
              [8, 9],
              [1, 0.9],
              [1, 0.8],
              [9, 11]])

plt.scatter(X[:, 0], X[:, 1], s=150, linewidths=5)
plt.show()

clf = KMeans(n_clusters=6)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_

colors = 10* ['g.', 'r.', 'c.', 'b.', 'k.']

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidth=10)
plt.show()



class K_Means:
    def __init__(self, k = 2, tol = 0.001, max_iter= 300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        
    def fit (self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i]= data[i]
        for i in range (self.max_iter):
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i]= []
            for featureset in X:
                distances = [np.linalg.norm(featureset -self.centroids [centroids]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
                
            prev_centroids = dict(self.centroids)
            for classification in self.classifications:
                pass
            self.centorids[classification] = np.avarage(self.classifications[classification], axis = 0)
            
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum (current_centroid-original_centroid/original_centroid*100.0) > self.tol:
                    optimized = False
                if optimized:
                    break
                
            
            
    def predict (self, data):
        
        distances = [np.linalg.norm(data - self.centroids [centroids]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
    
clf =K_Means()
clf.fit(X)
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid[0], clf.centroids[centroid][1]],
                marker = "o", color = 'k', s = 150, linewidths = 5)
    
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1],marker = "x", color = color , s =150, linewidths = 5)


unknowns = np.array[[1,2],
                    [9,3],
                    [2,3],
                    [2,9],
                    [1,2],
                    [3,3]]


for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker = "*", color = color [classification], s = 150, linewidths = 5)



plt.show()




























############################

df = pd.read_excel('titanic.xls')
print (df.head())
df.drop(['body','name'], 1, inplace = True)


df.convert_objects(convert_numeric = True)
df.fillna(0, inplace = True)


def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:       
        text_digit_vals = {}
        
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df [column].values
            unique_elements = set(column_contents)
            x = 0
            
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
                    
            df[column]= list(map(convert_to_int, df(column)))
            
    return df


df = handle_non_numerical_data()
print(df.head())
df.drop(['boat'], 1, inplace = True)


X = np.array(df.drop(['survival'], 1).astype(float))
#X = preprocessing.scale(X)
y = np.array(df['survival'])

clf = KMeans(n_cluster =2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct +=1
        
    label = clf.labels

####################################






                
                
                
