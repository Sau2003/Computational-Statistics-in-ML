import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

household_id=[1,2,3,4,5,6,7,8,9,10]
household_size=[2,4,4,1,3,5,6,4,7,2]
annual_income=[37000,49000,58000,68000,61000,64000,79000,89000,104000,95000]
no_pets=[0,0,1,3,2,2,1,1,1,0]

data=pd.DataFrame({'x':household_id,'y':household_size,'z':annual_income,'w':no_pets})

print("Data Information")
print(data.info())

print("Statistical description of the data")
print(data.describe())


# Visualization of household size 
plt.figure(figsize=(10,6))
sns.histplot(data['y'],bins=range(1,8),kde=True,color='red')
plt.title("Distribution of Household sizes")
plt.xlabel("Household Size")
plt.ylabel("Frequency")
plt.show()

# Visualization for annual income
plt.figure(figsize=(10,6))
sns.histplot(data['z'],bins=range(1,8),kde=True,color='orange')
plt.xtitle("Distribution of annual income")
plt.ytitle("Frequency")
plt.show()

# It can be observed from the dataset that the household size and annual income can be the related feature. So we need to find the relation between them  
plt.figure(figsize=(10,6))
sns.scatterplot(x='y',y='z',data=data,color='orange')
plt.title("Realtionship between household size and annual income")
plt.xlabel("Household size")
plt.ylabel("Annual income")
plt.show()

# # Visualize the distribution of the number of pets
plt.figure(figsize=(10,6))
sns.countplot(x='w',data=data,color="purple")
plt.title("Distribution of number of pets")
plt.xlabel("Number of pets")
plt.ylabel("Count")
plt.show()
