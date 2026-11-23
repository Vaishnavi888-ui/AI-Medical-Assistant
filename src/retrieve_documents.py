from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Locate the project and read the medical document
project_folder = Path(__file__).resolve().parent.parent
input_file = project_folder / "rag_documents" / "diabetes_who.txt"

text = input_file.read_text(encoding="utf-8")

# Re-create the same overlapping chunks
chunk_size = 500
overlap = 100
chunks = []

start = 0
while start < len(text):
    chunk = text[start:start + chunk_size].strip()
    if chunk:
        chunks.append(chunk)
    start += chunk_size - overlap

# Convert chunks into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
document_vectors = vectorizer.fit_transform(chunks)

print("Medical document retrieval system is ready!")
print(f"Total chunks indexed: {len(chunks)}")

# Ask the user for a question
query = input("\nEnter your medical question: ").strip()

if not query:
    print("Please enter a question.")
else:
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        document_vectors
    ).flatten()

    # Retrieve the three most relevant chunks
    top_indices = scores.argsort()[::-1][:3]

    print("\nMost relevant medical information:\n")

    for rank, index in enumerate(top_indices, start=1):
        print(f"\n--- Result {rank} (Similarity: {scores[index]:.3f}) ---")
        print(chunks[index])
