$path = ".\app.py"
$content = Get-Content $path -Raw

$old1 = @"
    context = "\n\n".join(sources)

    prompt = f"""
"@

$new1 = @"
    context = "\n\n".join(sources)

    recent_history = (chat_history or [])[-6:]
    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in recent_history
    )

    prompt = f"""
"@

$old2 = @"
User question:
{question}

WHO document excerpts:
"@

$new2 = @"
Recent conversation:
{history_text}

User's latest question:
{question}

WHO document excerpts:
"@

$old3 = @"
                            answer = generate_gemini_answer(
                                question,
                                sources
                            )
"@

$new3 = @"
                            answer = generate_gemini_answer(
                                question,
                                sources,
                                st.session_state["chat_history"][:-1]
                            )
"@

if (!$content.Contains($old1) -or !$content.Contains($old2) -or !$content.Contains($old3)) {
    Write-Host "A matching code section was not found. No changes made."
    exit
}

$content = $content.Replace($old1, $new1)
$content = $content.Replace($old2, $new2)
$content = $content.Replace($old3, $new3)

Set-Content -Path $path -Value $content -Encoding UTF8 -NoNewline
Write-Host "Conversation context added successfully."
