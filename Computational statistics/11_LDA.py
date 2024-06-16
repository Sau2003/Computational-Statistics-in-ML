import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
import seaborn as sns 

diam = [4.20, 3.76, 6.46, 3.15, 7.53, 10.42, 10.10, 9.04]
curv = [0.68, 0.78, 1.15, 0.27, 3.36, 2.48, 2.65, 3.23]
categ = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

data = pd.DataFrame({'x': diam, 'y': curv, 'category': categ})

# Visualizing the data
plt.figure(figsize=(10, 8))
sns.scatterplot(x='x', y='y', hue='category', data=data, color="red")
plt.title("Distribution of diameter vs curve")
plt.xlabel("Diameter")
plt.ylabel("Curvature")
plt.show()

lda = LinearDiscriminantAnalysis()
X = data[['x', 'y']]
y = data['category']
lda.fit(X, y)

# Visualize the LDA result
plt.figure(figsize=(8, 6))
sns.scatterplot(x='x', y='y', hue='category', data=data, palette='Set1', s=100, edgecolor="black")
plt.title("Linear Discriminant Analysis Result")
plt.xlabel("Diameter")
plt.ylabel("Curvature")

# Plot the decision regions
h = 0.02
x_min, x_max = X['x'].min() - 1, X['x'].max() + 1
y_min, y_max = X['y'].min() - 1, X['y'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.3)

# Plot original data points
sns.scatterplot(x='x', y='y', hue='category', data=data, palette='Set1', s=100, edgecolor="black")
plt.title("Linear Discriminant Analysis with Decision Regions")
plt.xlabel("Diameter")
plt.ylabel("Curvature")
plt.show()
