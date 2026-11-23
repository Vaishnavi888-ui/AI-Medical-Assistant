
import joblib
import pandas as pd

from pathlib import Path

# Locate saved model
project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "models" / "diabetes_model.pkl"

# Load trained model
model = joblib.load(model_path)

# Fictional sample patient data
sample_patient = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 80,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30
}])

# Make prediction
prediction = model.predict(sample_patient)[0]

probability = model.predict_proba(sample_patient)[0][1]

print("\n========== DIABETES RISK MODEL ==========")

if prediction == 1:
    print("Model prediction: Positive class (Outcome 1)")
else:
    print("Model prediction: Negative class (Outcome 0)")

print("Estimated model probability for positive class:",
      round(probability * 100, 2), "%")

print("\nNote: This is an educational ML demonstration.")
print("It is NOT a medical diagnosis or validated clinical risk estimate.")
