
from pathlib import Path

from app.ingestion import chunk_text


document_path = Path("data/sample.txt")
text = document_path.read_text(encoding="utf-8")

chunks = chunk_text(text)

print(f"Total chunks: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(chunk)
