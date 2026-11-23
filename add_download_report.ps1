$path = ".\app.py"
$content = Get-Content $path -Raw

$marker = '            st.subheader("Risk Assessment")'

$block = @"
            # Downloadable Prediction Report
            import io

            report = f'''
DIABETES PREDICTION REPORT
=========================

MODEL RESULT
Predicted Class: {'Positive' if prediction == 1 else 'Negative'}
Positive-Class Model Probability: {probability * 100:.2f}%

PATIENT MEASUREMENTS
{values.to_string(index=False)}

FEATURE EXPLANATION
'''

            for _, item in explanation.head(3).iterrows():
                direction = (
                    "Increased" if item["Contribution"] > 0
                    else "Decreased"
                )
                report += f"{item['Feature']}: {direction} the model score\n"

            report += '''

IMPORTANT MEDICAL DISCLAIMER
This is an educational AI prediction, not a medical diagnosis.
The model is not clinically validated. Consult a qualified
healthcare professional for medical advice and testing.
'''

            st.download_button(
                label="Download Prediction Report",
                data=report,
                file_name="diabetes_prediction_report.txt",
                mime="text/plain"
            )

"@

if ($content.Contains("Downloadable Prediction Report")) {
    Write-Host "Download report feature already exists."
} elseif ($content.Contains($marker)) {
    $content = $content.Replace($marker, $block + $marker)
    Set-Content -Path $path -Value $content -Encoding UTF8 -NoNewline
    Write-Host "Download report feature inserted successfully."
} else {
    Write-Host "Insertion point not found. No changes made."
}
