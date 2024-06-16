import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x=np.array([1,2,3,4,5]).reshape(-1,1)
y=np.array([2,3,4,5,6])

model=LinearRegression()
model.fit(x,y)

y_pred=model.predict(x)

plt.scatter(x,y,label="Data Points")
plt.plot(x,y_pred,color='red',label="linear Regression line")
plt.xlabel("Independent X")
plt.ylabel("Dependent Y")
plt.legend()
plt.show()

reg_coeff=model.coef_
intercept=model.intercept_

print(f"Slope is {reg_coeff}")
print(f"Intercept is {intercept}")