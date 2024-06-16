import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

X=[1,2,3,4,5]
Y=[3,6,7,8,11]

data=pd.DataFrame({'X':X,'Y':Y})

mean_x=data['X'].mean()
mean_y=data['Y'].mean()

std_dev_x=data['X'].std()
std_dev_y=data['Y'].std()

correlation=data['X'].corr(data['Y'])   # multi 

plt.scatter(data['X'],data['Y'])
plt.title("Scatter plot of x vs y")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

print(f"Mean is{mean_x}")
print(f"Mean is{mean_y}")
print(f"Mean is{std_dev_x}")
print(f"Mean is{std_dev_y}")
print(f"Mean is{correlation}")

summary_x=data.describe()
print(summary_x)