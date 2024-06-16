import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans

data=pd.DataFrame({'x':[4,5,10,4,3,11,14,6,10,12],
'y':[21,19,24,17,16,25,24,22,21,21]})

# combine x and y into 2d array
points=data.values
num_clusters=3

#Apply kmeans
kmeans=KMeans(n_clusters=num_clusters)
kmeans.fit(points)

# get cluster labels and centroids 
cluster_labels=kmeans.labels_
centroids=kmeans.cluster_centers_

# Add cluster labels to df
data['cluster']=cluster_labels


plt.figure(figsize=(8,6))
sns.scatterplot(x='x',y='y',data=data,hue='cluster',palette='viridis',s=100)
plt.scatter(centroids[:,0],centroids[:,1],marker='X',s=200,c='red',label='centroids')
plt.title("k-Means Clustering")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

