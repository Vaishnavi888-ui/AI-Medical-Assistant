
import pandas as pd
from pathlib import Path

# Locate the project folder
project_folder = Path(__file__).resolve().parent.parent
dataset_path = project_folder / "dataset" / "diabetes.csv"

# Load dataset
df = pd.read_csv(dataset_path)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== DIABETES OUTCOME COUNTS ==========")
print(df["Outcome"].value_counts())
