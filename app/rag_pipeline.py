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
from app.model_router import select_model
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
            "routing_ms": 0.0,
            "selected_model": None,
            "complexity": None,
            "routing_confidence": None,
        }
        return cached_response

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
                    "routing_ms": 0.0,
                    "selected_model": None,
                    "complexity": None,
                    "routing_confidence": None,
                },
            }

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
        semantic_lookup_ms = (
            perf_counter() - semantic_lookup_start
        ) * 1000

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
    routing_ms = 0.0
    selected_model = None
    complexity = None
    routing_confidence = None
    generation_ms = 0.0

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

        routing_start = perf_counter()
        routing_result = select_model(question)
        routing_ms = (perf_counter() - routing_start) * 1000

        selected_model = routing_result["model"]
        complexity = routing_result["complexity"]
        routing_confidence = routing_result["confidence"]

        generation_start = perf_counter()
        answer = generate_answer(
            question,
            context,
            model=selected_model,
        )
        generation_ms = (perf_counter() - generation_start) * 1000

    response = {
        "answer": answer,
        "sources": sources,
        "metrics": {
            "retrieval_ms": round(retrieval_ms, 2),
            "routing_ms": round(routing_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "end_to_end_ms": round(
                (perf_counter() - request_start) * 1000, 2
            ),
            "cache_lookup_ms": round(cache_lookup_ms, 2),
            "semantic_lookup_ms": round(semantic_lookup_ms, 2),
            "cache_hit": False,
            "semantic_cache_hit": False,
            "selected_model": selected_model,
            "complexity": complexity,
            "routing_confidence": (
                round(routing_confidence, 4)
                if routing_confidence is not None
                else None
            ),
        },
    }

    if cache_available:
        try:
            cache_response(_cache_client, cache_key, response)
        except (redis.exceptions.RedisError, TypeError, ValueError):
            pass

    
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
        pass

    return response
