$path = ".\app.py"
$content = Get-Content $path -Raw

$content = $content.Replace(
    'query_vector = vectorizer.transform([question])',
    @"
                previous_questions = [
                    message["content"]
                    for message in st.session_state["chat_history"][:-1]
                    if message["role"] == "user"
                ][-2:]

                retrieval_query = " ".join(previous_questions + [question])
                query_vector = vectorizer.transform([retrieval_query])
"@
)

$content = $content.Replace('for iin selected_indices', 'for i in selected_indices')

Set-Content -Path $path -Value $content -Encoding UTF8 -NoNewline
Write-Host "Follow-up retrieval and list comprehension fixes applied."
