
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Locate project folders
project_folder = Path(__file__).resolve().parent.parent

dataset_path = project_folder / "dataset" / "diabetes.csv"
model_path = project_folder / "models" / "diabetes_model.pkl"

# Load original dataset
df = pd.read_csv(dataset_path)

# Replace invalid zero measurements with missing values
zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[zero_as_missing] = df[zero_as_missing].replace(0, float("nan"))

# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])

# Create machine learning pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate model
print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("ROC-AUC Score:", round(roc_auc_score(y_test, y_prob), 4))

# Save trained model
joblib.dump(model, model_path)

print("\nModel saved successfully at:")
print(model_path)
