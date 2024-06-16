import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

y=[140,155,159,179,192,200,212,215]
x1=[60,62,67,70,71,72,75,78]
x2=[22,25,24,20,15,14,14,11]

data=pd.DataFrame({'x1':x1,'x2':x2,'y':y})

model=LinearRegression()
X=data[['x1','x2']]
y=data['y']
model.fit(X,y)
predictions=model.predict(X)

r2=r2_score(y,predictions)
print(r2)
print(f"The intercept is {model.intercept_}")
print(f"The coefficient is {model.coef_}")

plt.subplot(1,2,1)
sns.scatterplot(x='x1',y='y',data=data,color="blue")
plt.plot(x1,predictions,color="red",linewidth=2)
plt.title("Multiple linear regression x1 vs y")
plt.xlabel('x1')
plt.ylabel('y')

plt.subplot(1,2,2)
sns.scatterplot(x='x2',y='y',data=data,color="orange")
plt.plot(x2,predictions,color="purple",linewidth=2)
plt.title("Multiple linear regression: x2 vs y")
plt.xlabel('x2')
plt.ylabel('y')
plt.tight_layout()
plt.show()