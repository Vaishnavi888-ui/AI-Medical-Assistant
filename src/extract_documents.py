from pathlib import Path
from bs4 import BeautifulSoup

# Locate project folders
project_folder = Path(__file__).resolve().parent.parent
input_file = project_folder / "rag_documents" / "diabetes_who.html"
output_file = project_folder / "rag_documents" / "diabetes_who.txt"

# Read the downloaded HTML
with open(input_file, "r", encoding="utf-8") as file:
    html_content = file.read()

# Extract readable text
soup = BeautifulSoup(html_content, "html.parser")

# Remove unnecessary webpage elements
for element in soup(["script", "style", "nav", "footer", "header", "form", "noscript"]):
    element.decompose()

# Prefer the main article or page content
content = soup.find("article") or soup.find("main") or soup.body or soup

medical_text = content.get_text(separator="\n", strip=True)

# Save extracted text
output_file.write_text(medical_text, encoding="utf-8")

print("Text extraction completed!")
print(f"Output file: {output_file}")
print(f"Extracted characters: {len(medical_text)}")
print("\nPreview of extracted text:\n")
print(medical_text[:1000])
