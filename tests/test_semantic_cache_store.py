from app.vector_store import (
    create_vector_store,
    create_semantic_cache_collection,
    SEMANTIC_CACHE_COLLECTION,
    COLLECTION_NAME,
    VECTOR_SIZE,
)
from qdrant_client import models

client = create_vector_store()

try:
    create_semantic_cache_collection(client)
    create_semantic_cache_collection(client)

    assert client.collection_exists(COLLECTION_NAME)
    assert client.collection_exists(SEMANTIC_CACHE_COLLECTION)

    info = client.get_collection(SEMANTIC_CACHE_COLLECTION)
    config = info.config.params.vectors

    assert config.size == VECTOR_SIZE
    assert config.distance == models.Distance.COSINE

    print("Document collection preserved: PASS")
    print("Semantic cache collection created: PASS")
    print("Vector configuration verified: PASS")
    print("Repeated initialization: PASS")
finally:
    client.close()
