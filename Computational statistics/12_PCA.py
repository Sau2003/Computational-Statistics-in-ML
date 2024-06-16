import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA

diameter=[6.62,4.43,1.98,3.09,9.66,8.67,11.1,8.25]
weight=[21.40,21.01,20.71,20.26,16.47,15.28,16.10,15.87]
curvature=[0.56,0.78,0.53,0.89,2.37,2.89,2.69,3.47]

data=pd.DataFrame({'x':diameter,'y':weight,'z':curvature})

standardized_data=(data-data.mean())/data.std()

# Apply PCA

pca=PCA(n_components=2)
principal_components=pca.fit_transform(standardized_data)

# creating a dataframe with the principal components 
pca_df=pd.DataFrame(data=principal_components,columns=['PC1','PC2'])

#Concatenating the principal component with the original data
final_data=pd.concat([pca_df,data],axis=1)

#visualize the result
plt.figure(figsize=(10,6))
sns.scatterplot(x='PC1',y='PC2',data=final_data,color='blue')
plt.title("PCA Result")
plt.xlabel("Principal Component 1 ")
plt.ylabel("Principal Componenet 2")
plt.show()


