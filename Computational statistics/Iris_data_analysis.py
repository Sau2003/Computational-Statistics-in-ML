import pandas as pd
df=pd.read_csv("iris_csv.csv")
df.head()  # printing top 5 rows
df.info()  # printing inof

rows, columns = df.shape   # Shape fun for finding the nbr of rows and coulumns
print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")

print(df.describe())   # To get the quick summary of the datset like mean mode n all 
print(df.isnull().sum())   # To check whether our data contains any missimg value or not
data=df.drop_duplicates(subset="class",)
print(data)

df.value_counts("class")


#importing packages
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='class', data=df, )    # target column is the class col
plt.show()

sns.scatterplot(x='sepallength', y='sepalwidth',hue='class', data=df, )     # To print the scatterplot for comparing sepal lenght and sepal wi
# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

sns.scatterplot(x='petallength',y='petalwidth',hue='class',data=df,)
plt.legend(bbox_to_anchor=(1,1),loc=2)
plt.show()


#Let’s plot all the column’s relationships using a pairplot. It can be used for multivariate analysis.
sns.pairplot(df,hue='Species', height=2)
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(10,10))
 
axes[0,0].set_title("Sepal Length")
axes[0,0].hist(df['sepallength'], bins=7);
                                                           # HISTOGRAM
axes[0,1].set_title("Sepal Width")
axes[0,1].hist(df['sepalwidth'], bins=5);
 
axes[1,0].set_title("Petal Length")
axes[1,0].hist(df['petallength'], bins=6);
 
axes[1,1].set_title("Petal Width")
axes[1,1].hist(df['petalwidth'], bins=6);


plot = sns.FacetGrid(df, hue="class")
plot.map(sns.distplot, "sepallength").add_legend()
 
plot = sns.FacetGrid(df, hue="class")
plot.map(sns.distplot, "sepalwidth").add_legend()
 
plot = sns.FacetGrid(df, hue="class")
plot.map(sns.distplot, "petallength").add_legend()     # distplot is univar set of obs
 
plot = sns.FacetGrid(df, hue="class")
plot.map(sns.distplot, "petalwidth").add_legend()
 
plt.show()





# Assuming you have a DataFrame named 'df'
# Select numerical columns only
numeric_df = df.select_dtypes(include='number')

# Calculate the Pearson correlation matrix
correlation_matrix = numeric_df.corr(method='pearson')

# Print the correlation matrix
print(correlation_matrix)

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))  # Adjust the figure size if needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Pearson Correlation Heatmap')
plt.show()

 # FOR BOX PLOT
def graph(y):
    sns.boxplot(x="class", y=y, data=df)
 
plt.figure(figsize=(10,10))
     
# Adding the subplot at the specified
# grid position
plt.subplot(221)
graph('sepallength')
plt.subplot(222)
graph('sepalwidth')
plt.subplot(223)
graph('petallength')
plt.subplot(224)
graph('petalwidth')
plt.show()

sns.boxplot(x="sepalwidth", data=df)
plt.show()

import numpy as np
# Load the dataset
# IQR
Q1 = np.percentile(df['sepalwidth'], 25, interpolation='midpoint')
Q3 = np.percentile(df['sepalwidth'], 75, interpolation='midpoint')
IQR = Q3 - Q1

print("Old Shape: ", df.shape)

# Upper bound
upper = np.where(df['sepalwidth'] >= (Q3 + 1.5 * IQR))

# Lower bound
lower = np.where(df['sepalwidth'] <= (Q1 - 1.5 * IQR))

# Removing the Outliers
df.drop(upper[0], inplace=True)
df.drop(lower[0], inplace=True)

print("New Shape: ", df.shape)

#Visualize the boxplot after removing outliers
sns.boxplot(x='sepalwidth', data=df)
plt.show()
