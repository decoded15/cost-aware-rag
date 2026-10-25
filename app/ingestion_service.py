
import hashlib
from pathlib import Path

from app.embeddings import embed_chunks
from app.ingestion import chunk_text
from app.vector_store import store_chunks


def ingest_text(
    text: str,
    source: str,
    embedding_model,
    vector_store,
) -> dict:
    """Chunk, embed, and store a document in Qdrant."""

    if not text or not text.strip():
        raise ValueError("Document content cannot be empty.")

    if not source or not source.strip():
        raise ValueError("Document source cannot be empty.")

    normalized_source = Path(source).name
    document_id = hashlib.sha256(
        normalized_source.encode("utf-8")
    ).hexdigest()

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No text chunks were produced.")

    embeddings = embed_chunks(chunks, embedding_model)

    store_chunks(
        client=vector_store,
        chunks=chunks,
        embeddings=embeddings,
        document_id=document_id,
        source=normalized_source,
    )

    return {
        "document_id": document_id,
        "source": normalized_source,
        "chunks_created": len(chunks),
    }
