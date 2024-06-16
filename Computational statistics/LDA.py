import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# defining a function for calculation of stress level 
def calculate_stress_level(csv_file):
    df = pd.read_csv(csv_file)

    def calculate_stress(row):
        count = sum(row == "Yes")
        if count >= 4:
            return "High Stress"
        elif count == 3:
            return "Medium Stress"
        else:
            return "Low Stress"

    # Applying the calculate_stress function to each row for creation of new 'Stress Level' column
    df['Stress Level'] = df.apply(calculate_stress, axis=1)

    return df

# Loading the CSV file and calculation of stress levels
csv_file = 'stress.csv'
result_df = calculate_stress_level(csv_file)

# Create a dictionary to map long variable names to shorter labels
short_labels = {
    "Do you feel academic pressure?": "Academic Pressure",
    "Do you feel financial  stress?": "Financial Stress",
    "Are you getting enough sleep(7 - 8hrs per day)?": "Sleep Stress",
    "Do you feel stress due to poor Time Management?": "Time Management",
    "Do you feel stress due to lack of interaction in classroom?": "Interaction",
    "Do you feel stress due to the tight deadlines of college assignments and strict attendance?": "Attendance"
}

# Rename columns in the DataFrame with shorter labels
result_df = result_df.rename(columns=short_labels)

# Map "yes" and "no" to 1 and 0 for binary variables
binary_vars = ["Academic Pressure", "Financial Stress", "Sleep Stress", "Time Management", "Interaction", "Attendance"]
for var in binary_vars:
    result_df[var] = result_df[var].map({"Yes": 1, "No": 0})

# Extract features and target variable
X = result_df[binary_vars]
y = result_df['Stress Level']

# Apply Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components=2)  # Reducing to 2 dimensions for visualization
X_lda = lda.fit_transform(X, y)

# Combine LDA components with target variable for plotting
lda_df = pd.DataFrame(data=X_lda, columns=['LDA1', 'LDA2'])
lda_df['Stress Level'] = y.values

# Pairplot for visualization
sns.pairplot(lda_df, hue='Stress Level', palette='viridis')
plt.suptitle('Pairplot for LDA Components by Stress Level', y=1.02)
plt.show()
