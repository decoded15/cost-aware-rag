
from pathlib import Path

from app.ingestion import chunk_text
from app.embeddings import load_embedding_model, embed_chunks
from app.vector_store import (
    create_vector_store,
    store_chunks,
    search_chunks,
)

text = Path("data/sample.txt").read_text(encoding="utf-8")
chunks = chunk_text(text)

model = load_embedding_model()
embeddings = embed_chunks(chunks, model)

client = create_vector_store()
store_chunks(client, chunks, embeddings, document_id = "sample-document", source = "data/sample.txt")

question = "How can we reduce the cost of repeated AI responses?"
query_embedding = model.encode(
    question,
    normalize_embeddings=True,
).tolist()

results = search_chunks(client, query_embedding, limit=3)

for index, result in enumerate(results, start=1):
    print(f"\n--- Result {index} | Score: {result['score']:.4f} ---")
    print(result["text"])

client.close()
