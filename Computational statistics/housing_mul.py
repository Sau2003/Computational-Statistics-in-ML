# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # Read the raw data from the URL
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# # Combine every two rows to form the dataset
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]

# # Create a DataFrame with the combined data and target
# boston_df = pd.DataFrame(data, columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])
# boston_df["MEDV"] = target.astype(float)

# # Select the features (X) and target variable (y)
# X = boston_df.drop('MEDV', axis=1)
# y = boston_df['MEDV']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a linear regression model and fit it to the training data
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Display the intercept and regression coefficients
# intercept = model.intercept_
# coefficients = model.coef_

# print("Intercept:", intercept)
# print("Coefficients:", coefficients)

# # Make predictions on the testing set
# y_pred = model.predict(X_test)

# # Calculate and display the mean squared error
# mse = mean_squared_error(y_test, y_pred)
# print("\nMean Squared Error:", mse)

# # Create a scatter plot of predicted vs actual values with different colors
# plt.scatter(y_test, y_pred, c=['blue' if actual > predicted else 'red' for actual, predicted in zip(y_test, y_pred)])
# plt.xlabel("Actual Prices")
# plt.ylabel("Predicted Prices")
# plt.title("Actual Prices vs Predicted Prices")
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read the raw data from the URL
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Combine every two rows to form the dataset
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Create a DataFrame with the combined data and target
boston_df = pd.DataFrame(data, columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])
boston_df["MEDV"] = target.astype(float)

# Select the features (X) and target variable (y)
X = boston_df.drop('MEDV', axis=1)
y = boston_df['MEDV']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Display the intercept and regression coefficients
intercept = model.intercept_
coefficients = model.coef_

print("Intercept:", intercept)
print("Coefficients:", coefficients)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate and display the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

# Create a scatter plot of predicted vs actual values with different colors
plt.scatter(y_test, y_pred, c=['blue' if actual > predicted else 'red' for actual, predicted in zip(y_test, y_pred)])
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
