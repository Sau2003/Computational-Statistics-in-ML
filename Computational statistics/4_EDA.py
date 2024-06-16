import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

self_esteem=[20,27,30,32,50,55,62,64,77,80,85,90,100]
hrs_insta=[10,9,8,6,4,2,3,2,3,1,1,2,0]

data=pd.DataFrame({'x':self_esteem,'y': hrs_insta})

print("Data Information")
print(data.info())

print("Statistical Information")
print(data.describe())

#Visualization for self esteem
plt.figure(figsize=(10,6))
sns.histplot(data['x'],bins=range(1,8),kde=True,color="green")
plt.title("Distribution for Self esteem")
plt.xlabel("Self esteem")
plt.ylabel("frequency")
plt.show()

# Visualization for hrs at insta
plt.figure(figsize=(10,6))
sns.histplot(data['y'],bins=range(1,8),kde=True,color="blue")
plt.title("Distribution of Hours soen at Insta")
plt.xlabel("Hours at Insta")
plt.ylabel("Frequency")
plt.show()

# Visualization for the relationship between the Self esteem and hrs spen at Insta 
plt.figure(figsize=(10,8))
sns.scatterplot(x='x',y='y',data=data,color="red")
plt.title("Relationship between Self esteem and nbr of hrs at insta")
plt.xlabel("Self esteem")
plt.ylabel("Hours spend at Insta")
plt.show()

