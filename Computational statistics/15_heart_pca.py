import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the heart disease dataset from a CSV file
heart_data = pd.read_csv('heart.csv')

# Separate features (X) and target variable (y)
X = heart_data.drop('target', axis=1)
y = heart_data['target']

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_standardized)

# Variance explained by each principal component
explained_variance_ratio = pca.explained_variance_ratio_

# Plot the explained variance
plt.plot(np.cumsum(explained_variance_ratio))
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.show()

# Choose the number of components based on the plot or desired explained variance
num_components = 2  # You can choose a different number based on the plot

# Apply PCA with the selected number of components
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X_standardized)

# Create a DataFrame with PCA components
pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(num_components)])
pca_df['target'] = y

# Visualize the PCA result
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='target', data=pca_df, palette='tab10', s=20, edgecolor="None", alpha=0.7)
plt.title('Principal Component Analysis on Heart Disease Dataset')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.legend(title='Target', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
