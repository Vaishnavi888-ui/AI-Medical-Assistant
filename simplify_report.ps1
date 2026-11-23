$path = ".\app.py"
$content = Get-Content $path -Raw

$old = @"
            st.write("### Contribution Details")
            st.dataframe(
                explanation[["Feature", "Direction", "Contribution"]].round(4),
                hide_index=True,
                use_container_width=True
            )
"@

$new = @"
            st.write("### Simple Prediction Explanation")
            st.write(
                "Here are the three measurements that contributed most "
                "to the model's prediction:"
            )

            top_features = explanation.head(3)

            for _, item in top_features.iterrows():
                direction = (
                    "pushed the model score upward"
                    if item["Contribution"] > 0
                    else "pushed the model score downward"
                )

                feature_name = item["Feature"].replace(
                    "DiabetesPedigreeFunction",
                    "Family history indicator"
                )

                st.markdown(
                    f"- **{feature_name}:** This measurement {direction}."
                )

            st.info(
                "These explanations describe how the AI model used the "
                "measurements. They do not identify medical causes or "
                "confirm whether someone has diabetes."
            )
"@

if ($content.Contains($old)) {
    $content = $content.Replace($old, $new)
    Set-Content -Path $path -Value $content -Encoding UTF8 -NoNewline
    Write-Host "Prediction report simplified successfully."
} else {
    Write-Host "Could not find the exact Contribution Details section. No changes made."
}
