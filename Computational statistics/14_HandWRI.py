import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.datasets import fetch_openml

# Load the MNIST dataset with parser set to 'auto'
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype('float64')
y = mnist.target.astype('int64')

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a DataFrame with PCA components
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['label'] = y

# Visualize the PCA result
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='label', data=pca_df, palette='tab10', s=20, edgecolor="None", alpha=0.5)
plt.title('Principal Component Analysis on MNIST Dataset')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
