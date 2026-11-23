from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
input_file = project_folder / "rag_documents" / "diabetes_who.txt"
output_file = project_folder / "rag_documents" / "diabetes_chunks.txt"

text = input_file.read_text(encoding="utf-8")

chunk_size = 500
overlap = 100

chunks = []

start = 0
while start < len(text):
    end = start + chunk_size
    chunk = text[start:end].strip()

    if chunk:
        chunks.append(chunk)

    start += chunk_size - overlap

with open(output_file, "w", encoding="utf-8") as file:
    for index, chunk in enumerate(chunks, start=1):
        file.write(f"\n--- CHUNK {index} ---\n")
        file.write(chunk)
        file.write("\n")

print("Document chunking completed!")
print(f"Total chunks created: {len(chunks)}")
print(f"Saved to: {output_file}")
