$path = ".\app.py"
$content = Get-Content $path -Raw

$old1 = @"
                    sources = [
                        chunks[i]
                        for i in indices
                        if scores[i] > 0
                    ]
"@

$new1 = @"
                    selected_indices = [
                        i for i in indices if scores[i] > 0
                    ]

                    sources = [chunks[i] for i in selected_indices]
                    source_scores = [
                        float(scores[i]) for i in selected_indices
                    ]
"@

$old2 = @"
                            for number, passage in enumerate(
                                message["sources"], 1
                            ):

                                st.markdown(f"**Passage {number}**")
                                st.write(passage)
"@

$new2 = @"
                            source_scores = message.get("source_scores", [])

                            for number, passage in enumerate(
                                message["sources"], 1
                            ):
                                st.markdown(f"**WHO Source Passage {number}**")

                                if number <= len(source_scores):
                                    st.caption(
                                        f"Retrieval relevance score: {source_scores[number - 1]:.3f}"
                                    )

                                st.write(passage)
"@

$old3 = @"
                    "content": answer,
                    "sources": sources
"@

$new3 = @"
                    "content": answer,
                    "sources": sources,
                    "source_scores": source_scores if sources else []
"@

if (!$content.Contains($old1) -or !$content.Contains($old2) -or !$content.Contains($old3)) {
    Write-Host "A matching code section was not found. No changes made."
    exit
}

$content = $content.Replace($old1, $new1)
$content = $content.Replace($old2, $new2)
$content = $content.Replace($old3, $new3)

Set-Content -Path $path -Value $content -Encoding UTF8 -NoNewline
Write-Host "Source relevance scores added successfully."
