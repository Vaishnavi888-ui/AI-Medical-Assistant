
import pandas as pd
from pathlib import Path

# Locate project folders
project_folder = Path(__file__).resolve().parent.parent

input_path = project_folder / "dataset" / "diabetes.csv"
output_path = project_folder / "dataset" / "diabetes_cleaned.csv"

# Load original dataset
df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# Columns where zero represents an invalid or unrecorded measurement
zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace zero values with missing values
df[zero_as_missing] = df[zero_as_missing].replace(0, pd.NA)

# Fill missing measurements with each column's median
for column in zero_as_missing:
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)

# Save cleaned dataset
df.to_csv(output_path, index=False)

print("\nCleaned dataset shape:", df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nZero values after cleaning:")
print((df[zero_as_missing] == 0).sum())

print("\nCleaned dataset saved to:", output_path)
