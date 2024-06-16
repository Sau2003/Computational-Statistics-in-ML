import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns 

adv=[90,120,150,100,130]
sales=[1000,1300,1800,1200,1380]

data=pd.DataFrame({'x':adv,'y':sales})

X=np.array(adv).reshape(-1,1)
y=sales

model=LinearRegression()
model.fit(X,y)
predictions=model.predict(X)


# Visualizing the regression line 
plt.figure(figsize=(10,8))
sns.scatterplot(x='x',y='y',data=data,color="salmon")
plt.plot(adv,predictions,color="red",linewidth=2)
plt.title("Linear regression of Advertising vs sales")
plt.xlabel("Adv")
plt.ylabel("Sales")
plt.show()


print(f"The intercept is {model.intercept_}")
print(f"The coefficient is {model.coef_[0]}")

r2=r2_score(y,predictions)
print(f"The R2 is {r2}%")





