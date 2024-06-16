import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'your_dataset.csv' with the actual file path)
# Assuming the dataset has columns like 'Date', 'Consumption', 'Temperature', etc.
df = pd.read_csv('your_dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Check the basic information about the dataset
print(df.info())

# Descriptive statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize the distribution of electricity consumption
plt.figure(figsize=(12, 6))
sns.histplot(df['Consumption'], bins=30, kde=True, color='blue')
plt.title('Distribution of Electricity Consumption')
plt.xlabel('Consumption')
plt.ylabel('Frequency')
plt.show()

# Visualize the relationship between temperature and consumption
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Temperature', y='Consumption', data=df, color='green')
plt.title('Relationship between Temperature and Electricity Consumption')
plt.xlabel('Temperature')
plt.ylabel('Consumption')
plt.show()

# Visualize time trends in electricity consumption
df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' to datetime format
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Consumption', data=df, color='orange')
plt.title('Electricity Consumption Over Time')
plt.xlabel('Date')
plt.ylabel('Consumption')
plt.show()
