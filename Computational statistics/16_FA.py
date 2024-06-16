import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

# Load the BFI dataset from a CSV file
bfi_data = pd.read_csv('bfi.csv')

# Drop any rows with missing values
bfi_data = bfi_data.dropna()

# Select relevant columns for Factor Analysis
bfi_factors = bfi_data.iloc[:, 1:26]  # Assuming the relevant columns are from 1 to 26

# Kaiser-Meyer-Olkin (KMO) Test and Bartlett's Test of Sphericity
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

kmo_all, kmo_model = calculate_kmo(bfi_factors)
bartlett_test_statistic, bartlett_test_p_value = calculate_bartlett_sphericity(bfi_factors)

print("KMO Test Statistic:", kmo_model)
print("Bartlett's Test Statistic:", bartlett_test_statistic)
print("Bartlett's p-value:", bartlett_test_p_value)

# Perform Factor Analysis
n_factors = 5  # You can choose a different number of factors based on your requirements
fa = FactorAnalyzer(n_factors, rotation='varimax')
fa.fit(bfi_factors)

# Get the factor loadings
factor_loadings = fa.loadings_

# Visualize the factor loadings
plt.figure(figsize=(10, 6))
plt.imshow(factor_loadings, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Factor Loadings Heatmap')
plt.xlabel('Items')
plt.ylabel('Factors')
plt.show()
