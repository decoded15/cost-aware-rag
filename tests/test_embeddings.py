
from pathlib import Path

from app.ingestion import chunk_text
from app.embeddings import load_embedding_model, embed_chunks

text = Path("data/sample.txt").read_text(encoding="utf-8")
chunks = chunk_text(text)

model = load_embedding_model()
embeddings = embed_chunks(chunks, model)

print(f"Chunks: {len(chunks)}")
print(f"Embedding count: {len(embeddings)}")
print(f"Dimensions per embedding: {len(embeddings[0])}")
