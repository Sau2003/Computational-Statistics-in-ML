import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

potatoes = [0, 2, 9, 2]
vegetables = [4, 3, 4, 1]
meat = [0, 2, 10, 1]
icecream = [0, 8, 4, 0]
labels = ['disagree', 'neutral', 'agree', 'strongly disagree']

data = pd.DataFrame({'x': potatoes, 'y': vegetables, 'z': meat, 'w': icecream, 'l': labels})

mean_x = data['x'].mean()
mean_y = data['y'].mean()
mean_z = data['z'].mean()
mean_w = data['w'].mean()

std_dev_x = data['x'].std()
std_dev_y = data['y'].std()
std_dev_z = data['z'].std()
std_dev_w = data['w'].std()

# For correlation matrix
correlation_matrix = data[['x', 'y', 'z', 'w']].corr()

# For Pairplot
sns.pairplot(data, hue='l')
plt.suptitle("Scatter Plot Matrix")
plt.show()

# For heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation matrix")
plt.show()
