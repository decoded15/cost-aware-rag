from pathlib import Path
from uuid import uuid5, NAMESPACE_URL
from datetime import datetime, timezone
from uuid import uuid4
from qdrant_client import QdrantClient, models

STORAGE_PATH = Path("data/qdrant")
COLLECTION_NAME = "documents"
VECTOR_SIZE = 384
SEMANTIC_CACHE_COLLECTION = "semantic_cache"

def create_vector_store() -> QdrantClient:
    STORAGE_PATH.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(STORAGE_PATH))

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    return client

def create_semantic_cache_collection(
    client: QdrantClient,
) -> None:
    if not client.collection_exists(SEMANTIC_CACHE_COLLECTION):
        client.create_collection(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


def store_semantic_cache_entry(
    client: QdrantClient,
    question: str,
    embedding: list[float],
    response: dict,
    ttl_seconds: int = 600,
) -> str:
    if len(embedding) != VECTOR_SIZE:
        raise ValueError("Embedding has an unexpected vector size.")

    if ttl_seconds <= 0:
        raise ValueError("TTL must be greater than zero.")

    entry_id = str(uuid4())
    expires_at = int(datetime.now(timezone.utc).timestamp()) + ttl_seconds

    client.upsert(
        collection_name=SEMANTIC_CACHE_COLLECTION,
        points=[
            models.PointStruct(
                id=entry_id,
                vector=embedding,
                payload={
                    "question": question,
                    "response": response,
                    "expires_at": expires_at,
                },
            )
        ],
        wait=True,
    )

    return entry_id


def search_semantic_cache(
    client: QdrantClient,
    query_embedding: list[float],
    similarity_threshold: float = 0.90,
) -> dict | None:
    if len(query_embedding) != VECTOR_SIZE:
        raise ValueError("Query embedding has an unexpected vector size.")

    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("Similarity threshold must be between -1 and 1.")

    results = client.query_points(
        collection_name=SEMANTIC_CACHE_COLLECTION,
        query=query_embedding,
        limit=5,
        with_payload=True,
    ).points

    now = int(datetime.now(timezone.utc).timestamp())

    for result in results:
        payload = result.payload or {}

        if payload.get("expires_at", 0) <= now:
            continue

        if result.score < similarity_threshold:
            continue

        return {
            "question": payload["question"],
            "response": payload["response"],
            "score": result.score,
        }

    return None

def store_chunks(
    client: QdrantClient,
    chunks: list[str],
    embeddings: list[list[float]],
    document_id: str,
    source: str,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have a corresponding embedding.")

    if not document_id.strip():
        raise ValueError("document_id cannot be empty.")

    # Find chunks already stored for this document.
    existing_points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        ),
        limit=10000,
        with_payload=False,
        with_vectors=False,
    )

    old_ids = {str(point.id) for point in existing_points}

    points = []
    new_ids = set()

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(
            uuid5(NAMESPACE_URL, f"{document_id}:{index}")
        )
        new_ids.add(point_id)

        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk,
                    "document_id": document_id,
                    "source": source,
                    "chunk_index": index,
                },
            )
        )

    # Upsert replaces existing points with the same IDs.
    if points:
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    # Remove chunks left over from an older, longer document.
    stale_ids = old_ids - new_ids

    if stale_ids:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=list(stale_ids),
            wait=True,
        )


def delete_document(
    client: QdrantClient,
    document_id: str,
) -> None:
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        ),
        wait=True,
    )


def search_chunks(
    client: QdrantClient,
    query_embedding: list[float],
    limit: int = 3,
) -> list[dict]:
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit,
        with_payload=True,
    ).points

    return [
        {
            "text": result.payload["text"],
            "source": result.payload["source"],
            "document_id": result.payload["document_id"],
            "chunk_index": result.payload["chunk_index"],
            "score": result.score,
        }
        for result in results
    ]
