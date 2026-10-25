
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
from app.generation import generate_answer
from app.vector_store import create_vector_store, search_chunks


_embedding_model = load_embedding_model()
_vector_store = create_vector_store()
_cache_client = get_cache_client()


def ask_question(question: str) -> dict:
    request_start = perf_counter()

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # 1. Check the exact-query cache first.
    cache_lookup_start = perf_counter()
    cache_key = make_cache_key(question)
    cached_response = None
    cache_available = True

    try:
        cached_response = get_cached_response(
            _cache_client,
            cache_key,
        )
    except (redis.exceptions.RedisError, ValueError, TypeError, json.JSONDecodeError):
        # Cache failure must not prevent RAG from working.
        cache_available = False

    cache_lookup_ms = (
        perf_counter() - cache_lookup_start
    ) * 1000

    if cached_response is not None:
        cached_response["metrics"] = {
            **cached_response.get("metrics", {}),
            "end_to_end_ms": round(
                (perf_counter() - request_start) * 1000, 2
            ),
            "cache_lookup_ms": round(cache_lookup_ms, 2),
            "cache_hit": True,
        }
        return cached_response

    # 2. Cache miss: run the existing RAG pipeline.
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

    retrieval_ms = (
        perf_counter() - retrieval_start
    ) * 1000

    if not results:
        response = {
            "answer": (
                "I couldn't find relevant information "
                "in the stored documents."
            ),
            "sources": [],
            "metrics": {
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": 0.0,
                "end_to_end_ms": 0.0,
            },
        }
    else:
        context_parts = []
        sources = []

        for result in results:
            source = result["source"]
            chunk_index = result["chunk_index"]
            text = result["text"]

            context_parts.append(
                f"[Source: {source}, chunk: {chunk_index}]\n{text}"
            )

            sources.append({
                "source": source,
                "chunk_index": chunk_index,
                "score": result["score"],
            })

        context = "\n\n".join(context_parts)

        generation_start = perf_counter()
        answer = generate_answer(question, context)
        generation_ms = (
            perf_counter() - generation_start
        ) * 1000

        response = {
            "answer": answer,
            "sources": sources,
            "metrics": {
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "end_to_end_ms": 0.0,
            },
        }

    response["metrics"]["end_to_end_ms"] = round(
        (perf_counter() - request_start) * 1000, 2
    )
    response["metrics"]["cache_lookup_ms"] = round(
        cache_lookup_ms, 2
    )
    response["metrics"]["cache_hit"] = False

    # 3. Save the complete response for the next request.
    if cache_available:
        try:
            cache_response(
                _cache_client,
                cache_key,
                response,
            )
        except (redis.exceptions.RedisError, TypeError, ValueError):
            # Return the valid RAG answer even if caching fails.
            pass

    return response
