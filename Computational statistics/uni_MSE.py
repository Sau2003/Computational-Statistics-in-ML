import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = [10, 15, 20, 25, 30]
data_series = pd.Series(data)
# Mean (average)
mean = data_series.mean()

# Median (middle value)
median = data_series.median()

# Mode (most frequent value)
mode = data_series.mode().values[0]

# Standard deviation
std_dev = data_series.std()

# Variance
variance = data_series.var()

# Range (max - min)
range_val = data_series.max() - data_series.min()
plt.hist(data_series, bins=5, edgecolor='k')
plt.title("Histogram of Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")
print(f"Range: {range_val}")
