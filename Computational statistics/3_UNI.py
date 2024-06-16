import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

temp=[20,25,35,43]
ice_cream_sales=[2000,2500,5000,7800]

data=pd.DataFrame({'x':temp,'y':ice_cream_sales})\

# Univariate analysis for the Temprature feature
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.histplot(data['x'],bins=10,kde=True,color="green")
plt.title("Distribution of the temprature")
plt.xlabel("Temprature")
plt.ylabel("Frequency")
plt.subplot(1,2,2)
sns.boxplot(x=data['x'],color="green")
plt.title("Boxplot of temperature")
plt.tight_layout
plt.show()

# Univariate Analysis for Icecream 
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.histplot(data['y'],bins=10,kde=True,color='orange')
plt.title("Distribution of Icecream")
plt.xlabel("Icecream")
plt.ylabel("Frequency")
plt.subplot(1,2,2)
sns.boxplot(data['y'],color="orange")
plt.title("Boxplot for Icecream")
plt.tight_layout
plt.show()

# Scatterplot for the relationship between temperature and the Icecream 
plt.figure(figsize=(8,6))
sns.scatterplot(x='x',y='y',data=data,color='salmon')
plt.title("Scatter Plot of temperature vs Icecream ")
plt.xlabel("temperature")
plt.ylabel("Ice cream Sales")
plt.show()

