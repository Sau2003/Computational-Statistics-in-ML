import pandas as pd  # to read csv files 
import numpy as np  # for multidimensiuonal and single dim array elements
import matplotlib.pyplot as plt  # for plotting plots 
import seaborn as sns  # for stastical graphs 

dataset=pd.read_csv("Advertising.csv")
dataset.head()  # for top 5 rows
print(dataset)
dataset.drop(columns=['Radio', 'Newspaper'], inplace = True)  # rem radio and newspaper 
dataset.head()
print(dataset)
x = dataset[['TV']]   # fitting the values 
y = dataset['Sales']

#Splitting the dataset
from sklearn.model_selection import train_test_split # It is used for splitting data arrays into two subsets: for training data 
# and testing data.usually good to keep 70% of the data in your train dataset and the rest 30% in your test dataset.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
#Random==== This parameter controls the shuffling applied to the data before
#  applying the split. Pass an int for reproducible output across multiple function calls.
# test_size: This parameter specifies the size of the testing dataset. The default state suits the training size. 
# It will be set to 0.25 if the training size is set to default.


# Fitting the Linear Regression model
from sklearn.linear_model import LinearRegression
slr = LinearRegression()  # Now slr is the instance of linear reg
slr.fit(x_train, y_train)

#Intercept and Coefficient
print("Intercept: ", slr.intercept_)
print("Coefficient: ", slr.coef_)   #  Sales = 6.948 + 0.054 * TV

#Prediction of test set
y_pred_slr= slr.predict(x_test)
#Predicted values
print("Prediction for test set: {}".format(y_pred_slr))



#Actual value and the predicted value
slr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_slr})
slr_diff.head()


#Line of best fit
plt.scatter(x_test,y_test)
plt.plot(x_test, y_pred_slr, 'Red')
plt.show()


#Model Evaluation
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_slr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_slr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_slr))
print('R squared: {:.2f}'.format(slr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


#from sklearn import metrics: It provides metrics for evaluating the model.

#R Squared: R Square is the coefficient of determination. It tells us how many points fall on the regression line. The value of R Square is 81.10, which indicates that 81.10% of the data fit the regression model.

#Mean Absolute Error: Mean Absolute Error is the absolute difference between the actual or true values and the predicted values. The lower the value, the better is the model’s performance. A mean absolute error of 0 means that your model is a perfect predictor of the outputs. The mean absolute error obtained for this particular model is 1.648, which is pretty good as it is close to 0.

#Mean Square Error: Mean Square Error is calculated by taking the average of the square of the difference between the original and predicted values of the data. The lower the value, the better is the model’s performance. The mean square error obtained for this particular model is 4.077, which is pretty good.

#Root Mean Square Error: Root Mean Square Error is the standard deviation of the errors which occur when a prediction is made on a dataset. This is the same as Mean Squared Error, but the root of the value is considered while determining the accuracy of the model. The lower the value, the better is the model’s performance. The root mean square error obtained for this particular model is 2.019, which is pretty good.

#Conclusion

#The Simple Linear Regression model performs well as 81.10% of the data fit the regression model. Also, the mean absolute error, mean square error, and the root mean square error are less.