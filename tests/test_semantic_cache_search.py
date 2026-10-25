
from app.semantic_embeddings import (
    load_semantic_embedding_model,
    embed_semantic_question,
)
from app.vector_store import (
    create_vector_store,
    create_semantic_cache_collection,
    store_semantic_cache_entry,
    search_semantic_cache,
)

QUESTION = "How can I reduce repeated LLM calls?"
ANSWER = "Caching lets us reuse previous responses and avoid unnecessary LLM calls."


def main():
    client = create_vector_store()

    try:
        create_semantic_cache_collection(client)

        model = load_semantic_embedding_model()
        question_embedding = embed_semantic_question(QUESTION, model)

        store_semantic_cache_entry(
            client=client,
            question=QUESTION,
            embedding=question_embedding,
            response=ANSWER,
            ttl_seconds=600,
        )

        test_queries = [
            QUESTION,
            "How do I avoid calling an LLM again for the same question?",
            "What is the capital of France?",
        ]

        for query in test_queries:
            query_embedding = embed_semantic_question(query, model)

            result = search_semantic_cache(
                client=client,
                query_embedding=query_embedding,
                similarity_threshold=0.80,
            )

            if result:
                print(f"\nQUERY: {query}")
                print(f"RESULT: HIT | Score: {result['score']:.4f}")
                print(f"CACHED ANSWER: {result['response']}")
            else:
                print(f"\nQUERY: {query}")
                print("RESULT: MISS")

    finally:
        client.close()


if __name__ == "__main__":
    main()
