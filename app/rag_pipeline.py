
import json
from time import perf_counter

import redis

from app.cache import (
    get_cache_client,
    get_cached_response,
    cache_response,
    make_cache_key,
)
from app.embeddings import load_embedding_model
from app.semantic_embeddings import (
    load_semantic_embedding_model,
    embed_semantic_question,
)
from app.semantic_safety import has_conflicting_intent
from app.generation import generate_answer
from app.vector_store import (
    create_vector_store,
    create_semantic_cache_collection,
    search_chunks,
    search_semantic_cache,
    store_semantic_cache_entry,
)


SEMANTIC_SIMILARITY_THRESHOLD = 0.80
SEMANTIC_CACHE_TTL_SECONDS = 600

_embedding_model = load_embedding_model()
_semantic_embedding_model = load_semantic_embedding_model()
_vector_store = create_vector_store()
create_semantic_cache_collection(_vector_store)
_cache_client = get_cache_client()


def ask_question(question: str) -> dict:
    request_start = perf_counter()

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # 1. Exact cache: cheapest lookup first.
    cache_lookup_start = perf_counter()
    cache_key = make_cache_key(question)
    cache_available = True
    cached_response = None

    try:
        cached_response = get_cached_response(_cache_client, cache_key)
    except (
        redis.exceptions.RedisError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):
        cache_available = False

    cache_lookup_ms = (perf_counter() - cache_lookup_start) * 1000

    if cached_response is not None:
        cached_response["metrics"] = {
            **cached_response.get("metrics", {}),
            "end_to_end_ms": round(
                (perf_counter() - request_start) * 1000, 2
            ),
            "cache_lookup_ms": round(cache_lookup_ms, 2),
            "cache_hit": True,
            "semantic_cache_hit": False,
        }
        return cached_response

    # 2. Semantic cache: only after an exact-cache miss.
    semantic_lookup_start = perf_counter()
    semantic_lookup_ms = 0.0

    try:
        semantic_embedding = embed_semantic_question(
            question, _semantic_embedding_model
        )

        semantic_result = search_semantic_cache(
            client=_vector_store,
            query_embedding=semantic_embedding,
            similarity_threshold=SEMANTIC_SIMILARITY_THRESHOLD,
        )

        semantic_lookup_ms = (
            perf_counter() - semantic_lookup_start
        ) * 1000

        if (
            semantic_result is not None
            and not has_conflicting_intent(
                question, semantic_result["question"]
            )
        ):
            response = {
                "answer": semantic_result["response"],
                "sources": [],
                "metrics": {
                    "retrieval_ms": 0.0,
                    "generation_ms": 0.0,
                    "end_to_end_ms": round(
                        (perf_counter() - request_start) * 1000, 2
                    ),
                    "cache_lookup_ms": round(cache_lookup_ms, 2),
                    "semantic_lookup_ms": round(semantic_lookup_ms, 2),
                    "cache_hit": False,
                    "semantic_cache_hit": True,
                    "semantic_similarity": round(
                        semantic_result["score"], 4
                    ),
                },
            }

            # Store the result under this exact wording too.
            if cache_available:
                try:
                    cache_response(_cache_client, cache_key, response)
                except (
                    redis.exceptions.RedisError,
                    TypeError,
                    ValueError,
                ):
                    pass

            return response

    except Exception:
        # Semantic cache failure must never block normal RAG.
        semantic_lookup_ms = (
            perf_counter() - semantic_lookup_start
        ) * 1000

    # 3. Both caches missed: run normal document retrieval.
    query_embedding = _embedding_model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    retrieval_start = perf_counter()
    results = search_chunks(
        client=_vector_store,
        query_embedding=query_embedding,
        limit=3,
    )
    retrieval_ms = (perf_counter() - retrieval_start) * 1000

    sources = []
    if not results:
        answer = (
            "I couldn't find relevant information "
            "in the stored documents."
        )
    else:
        context_parts = []

        for result in results:
            source = result["source"]
            chunk_index = result["chunk_index"]

            context_parts.append(
                f"[Source: {source}, chunk: {chunk_index}]\n"
                f"{result['text']}"
            )
            sources.append({
                "source": source,
                "chunk_index": chunk_index,
                "score": result["score"],
            })

        context = "\n\n".join(context_parts)

        generation_start = perf_counter()
        answer = generate_answer(question, context)
        generation_ms = (perf_counter() - generation_start) * 1000

    if not results:
        generation_ms = 0.0

    response = {
        "answer": answer,
        "sources": sources,
        "metrics": {
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "end_to_end_ms": round(
                (perf_counter() - request_start) * 1000, 2
            ),
            "cache_lookup_ms": round(cache_lookup_ms, 2),
            "semantic_lookup_ms": round(semantic_lookup_ms, 2),
            "cache_hit": False,
            "semantic_cache_hit": False,
        },
    }

    # 4. Cache the generated answer for future requests.
    if cache_available:
        try:
            cache_response(_cache_client, cache_key, response)
        except (redis.exceptions.RedisError, TypeError, ValueError):
            pass

    # Store the question and answer for semantic reuse.
    # Never mix this BGE vector with MiniLM document vectors.
    try:
        semantic_embedding = embed_semantic_question(
            question, _semantic_embedding_model
        )
        store_semantic_cache_entry(
            client=_vector_store,
            question=question,
            embedding=semantic_embedding,
            response=answer,
            ttl_seconds=SEMANTIC_CACHE_TTL_SECONDS,
        )
    except Exception:
        # A semantic-cache write failure doesn't invalidate the answer.
        pass

    return response
