import pandas as pd
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler

# Load your car price dataset
car_data = pd.read_csv('carprice.csv')
print(car_data.head())
print(car_data.info())

# Identify and handle missing values
car_data = car_data.replace('?', pd.NA)
car_data = car_data.dropna()  # or use fillna() to fill missing values

# Convert non-numeric columns to numeric (if needed)
car_data_numeric = car_data.apply(pd.to_numeric, errors='coerce')
car_data_numeric = car_data_numeric.dropna()

# Standardize the numerical variables
scaler = StandardScaler()
car_data_scaled = scaler.fit_transform(car_data_numeric)

# Factor Analysis
n_factors = 2
fa = FactorAnalyzer(n_factors, rotation='varimax')
fa.fit(car_data_scaled)

# Get factor loadings
factor_loadings = pd.DataFrame(fa.loadings_, index=car_data_numeric.columns)

# Get eigenvalues and variance explained
eigenvalues, variance_explained = fa.get_eigenvalues()

print("Factor Loadings:")
print(factor_loadings)

print("\nEigenvalues:")
print(eigenvalues)

print("\nVariance Explained:")
print(variance_explained)
