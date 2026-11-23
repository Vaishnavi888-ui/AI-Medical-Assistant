$path = ".\app.py"
$content = Get-Content $path -Raw
$marker = '            st.subheader("Risk Assessment")'

$block = @"
            # Explainable AI: Feature Contributions
            st.divider()
            st.subheader("Why did the model make this prediction?")

            transformed_values = model[:-1].transform(values)
            classifier = model.named_steps["classifier"]
            contributions = transformed_values[0] * classifier.coef_[0]

            explanation = pd.DataFrame({
                "Feature": values.columns,
                "Contribution": contributions
            })

            explanation["Direction"] = explanation["Contribution"].apply(
                lambda x: "Increases model score" if x > 0
                else "Decreases model score"
            )

            explanation["Absolute Impact"] = explanation["Contribution"].abs()
            explanation = explanation.sort_values("Absolute Impact", ascending=False)

            st.caption("These are mathematical model contributions, not medical causes.")
            st.write("### Feature Contribution Chart")
            st.bar_chart(explanation.set_index("Feature")["Contribution"])

            st.write("### Contribution Details")
            st.dataframe(
                explanation[["Feature", "Direction", "Contribution"]].round(4),
                hide_index=True,
                use_container_width=True
            )

"@

if ($content.Contains("Explainable AI: Feature Contributions")) {
    Write-Host "Already inserted. No changes made."
} elseif ($content.Contains($marker)) {
    $content = $content.Replace($marker, $block + $marker)
    Set-Content -Path $path -Value $content -Encoding UTF8 -NoNewline
    Write-Host "Explainable AI inserted successfully."
} else {
    Write-Host "Insertion point not found. No changes made."
}
