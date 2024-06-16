import numpy as np # used for working with arr
import pandas as pd # used for working with datasets
import matplotlib.pyplot as plt  # For graph plotting
import seaborn as sns  # data visualization library 
from sklearn.model_selection import train_test_split # implements ML and statistical model
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Custom input data 
x = [1.7,1.5,2.8,5,1.3,2.2,1.3]
y = [368,340,665,954,331,556,376]

# Reshaping  input data to 2D arrays
x = np.array(x).reshape(-1, 1)
y = np.array(y)

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# Fitting the Linear Regression model
slr = LinearRegression()
slr.fit(x_train, y_train)

# For Intercept and Coefficient
print("Intercept: ", slr.intercept_)
print("Coefficient: ", slr.coef_)

# Prediction of test set
y_pred_slr = slr.predict(x_test)

# Actual value and the predicted value
slr_diff = pd.DataFrame({'Actual value is ': y_test, 'and Predicted value': y_pred_slr})

# # best fit line
# plt.scatter(x_test, y_test, label='Actual Points')
# plt.plot(x_test, y_pred_slr, 'Red', label='Predicted Line')
# plt.xlabel('X values')
# plt.ylabel('Y values')
# plt.legend()
# plt.show()

# Predicting future values
x_future = np.array([4.5]).reshape(-1, 1) 
y_future = slr.predict(x_future)
print('Predicted future values:', y_future)


