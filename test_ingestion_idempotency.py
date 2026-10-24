
from pathlib import Path

from app.ingestion import chunk_text
from app.embeddings import load_embedding_model, embed_chunks
from app.vector_store import (
    COLLECTION_NAME,
    create_vector_store,
    store_chunks,
    delete_document,
)

document_path = Path("data/sample.txt")
text = document_path.read_text(encoding="utf-8")

chunks = chunk_text(text)
model = load_embedding_model()
embeddings = embed_chunks(chunks, model)

client = create_vector_store()
document_id = "sample-document"

try:
    store_chunks(
        client, chunks, embeddings,
        document_id, str(document_path),
    )

    first_count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    store_chunks(
        client, chunks, embeddings,
        document_id, str(document_path),
    )

    second_count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print(f"After first ingestion: {first_count}")
    print(f"After second ingestion: {second_count}")
    assert first_count == second_count, "Duplicate points detected!"

    # Test an update that produces fewer chunks.
    shorter_chunks = chunks[:2]
    shorter_embeddings = embed_chunks(shorter_chunks, model)

    store_chunks(
        client, shorter_chunks, shorter_embeddings,
        document_id, str(document_path),
    )

    updated_count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print(f"After shorter update: {updated_count}")
    assert updated_count == 2, "Stale chunks were not removed!"

    delete_document(client, document_id)

    final_count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print(f"After document deletion: {final_count}")
    assert final_count == 0, "Document vectors remain!"

    print("All ingestion tests passed.")

finally:
    client.close()
