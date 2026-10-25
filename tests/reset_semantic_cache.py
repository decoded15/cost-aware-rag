
from app.vector_store import (
    create_vector_store,
    SEMANTIC_CACHE_COLLECTION,
)

client = create_vector_store()

try:
    if client.collection_exists(SEMANTIC_CACHE_COLLECTION):
        client.delete_collection(SEMANTIC_CACHE_COLLECTION)
        print("Old semantic-cache collection deleted.")

    print("Document collection preserved.")
finally:
    client.close()
