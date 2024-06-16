import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data={'x1':[1,2,3,4,5],'x2':[23,46,77,51,34],'y':[2,3,4,5,6]}
df=pd.DataFrame(data)

X=df[['x1','x2']]
Y=df['y']

model=LinearRegression()
model.fit(X,Y)

Coeff=model.coef_
Inter=model.intercept_
print(f"Coeff is{Coeff}")
print(f"Intercept is {Inter}")