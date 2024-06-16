import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Apply LDA
X = iris.data
y = iris.target
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Create a DataFrame with LDA components
lda_df = pd.DataFrame(data=X_lda, columns=['LD1', 'LD2'])
lda_df['species'] = iris_df['species']

# Visualize the LDA result
plt.figure(figsize=(10, 8))
sns.scatterplot(x='LD1', y='LD2', hue='species', data=lda_df, palette='viridis', s=100, edgecolor="black")
plt.title('Linear Discriminant Analysis on Iris Dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.show()
