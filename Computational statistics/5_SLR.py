import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

TV = [42, 34, 25, 35, 37, 38, 31, 33, 19, 29, 38, 28, 29, 36, 18]
overweight = [18, 6, 0, -1, 13, 14, 7, 7, -9, 8, 8, 5, 3, 14, -7]

data = pd.DataFrame({'TV': TV, 'overweight': overweight})

# Reshape TV to a 2D array
X = np.array(TV).reshape(-1, 1)

model = LinearRegression()
model.fit(X, overweight)

predictions = model.predict(X)

# Visualize the regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TV', y='overweight', data=data, color="green")
plt.plot(TV, predictions, color="red", linewidth=2)
plt.title("Simple Linear Regression: TV with Overweight")
plt.xlabel("TV")
plt.ylabel("Overweight")
plt.show()

# Print the regression equation coefficients
print(f'Intercept: {model.intercept_}')
print(f'Coefficient: {model.coef_[0]}')
