
from app.embeddings import load_embedding_model
from app.vector_store import (
    create_vector_store,
    create_semantic_cache_collection,
    store_semantic_cache_entry,
)
from qdrant_client import models


EVAL_COLLECTION = "semantic_cache_eval"

CASES = [
    (
        "exact",
        "How can I reduce repeated LLM calls?",
    ),
    (
        "paraphrase",
        "How do I avoid calling an LLM again for the same question?",
    ),
    (
        "paraphrase",
        "How can caching prevent repeated language model requests?",
    ),
    (
        "related",
        "How can I reduce the cost of repeated AI responses?",
    ),
    (
        "unrelated",
        "What is the capital of France?",
    ),
]


model = load_embedding_model()
client = create_vector_store()

try:
    # Use a dedicated evaluation collection.
    if client.collection_exists(EVAL_COLLECTION):
        client.delete_collection(EVAL_COLLECTION)

    client.create_collection(
        collection_name=EVAL_COLLECTION,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
    )

    original_question = "How can I reduce repeated LLM calls?"
    original_embedding = model.encode(
        original_question,
        normalize_embeddings=True,
    ).tolist()

    client.upsert(
        collection_name=EVAL_COLLECTION,
        points=[
            models.PointStruct(
                id="8a3f6e12-7c4b-4d91-9e25-2b6f0c8d41a7",
                vector=original_embedding,
                payload={"question": original_question},
            )
        ],
        wait=True,
    )

    print("\n===== CLEAN SEMANTIC EVALUATION =====")

    for category, question in CASES:
        query_embedding = model.encode(
            question,
            normalize_embeddings=True,
        ).tolist()

        results = client.query_points(
            collection_name=EVAL_COLLECTION,
            query=query_embedding,
            limit=1,
            with_payload=True,
        ).points

        score = results[0].score if results else None

        print(f"\nCategory: {category}")
        print(f"Question: {question}")
        print(
            f"Similarity: {score:.4f}"
            if score is not None
            else "Similarity: no candidate"
        )

finally:
    if client.collection_exists(EVAL_COLLECTION):
        client.delete_collection(EVAL_COLLECTION)

    client.close()
