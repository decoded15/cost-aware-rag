import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Cost-Aware RAG Gateway",
    page_icon="⚡",
    layout="wide",
)

st.title("Cost-Aware RAG Gateway")
st.caption(
    "Local-first RAG · Semantic caching · ML-powered model routing"
)

# Sidebar: API status, document ingestion, and components.
with st.sidebar:
    st.header("Gateway status")

    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        if health.ok:
            st.success("FastAPI connected")
        else:
            st.warning(f"API returned {health.status_code}")
    except requests.RequestException:
        st.error("FastAPI unavailable")

    st.divider()
    st.header("📄 Add documents")
    st.caption("Upload a UTF-8 text file to your knowledge base.")

    uploaded_file = st.file_uploader(
        "Choose a text document",
        type=["txt"],
        key="document_uploader",
    )

    if st.button(
        "Upload and process",
        disabled=uploaded_file is None,
        use_container_width=True,
    ):
        try:
            with st.spinner("Chunking, embedding, and storing document..."):
                response = requests.post(
                    f"{API_URL}/ingest",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "text/plain",
                        )
                    },
                    timeout=(5, 600),
                )
                response.raise_for_status()
                result = response.json()

            st.success("Document ingested successfully!")
            st.write(f"**File:** {result.get('source', uploaded_file.name)}")
            st.write(f"**Chunks created:** {result.get('chunks_created', 0)}")
            st.caption(
                "You can now ask questions about this document in the chat."
            )

        except requests.Timeout:
            st.error("Ingestion timed out. Check the backend and try again.")
        except requests.HTTPError as exc:
            detail = "The API rejected the document."
            try:
                detail = exc.response.json().get("detail", detail)
            except (ValueError, AttributeError):
                pass
            st.error(f"Ingestion failed: {detail}")
        except requests.RequestException:
            st.error(
                "Couldn't reach FastAPI. Check that the backend is running."
            )

    st.divider()
    st.markdown("**Active components**")
    st.markdown("- Qdrant — document retrieval")
    st.markdown("- Valkey — exact response cache")
    st.markdown("- BGE — semantic cache")
    st.markdown("- Hugging Face — query classifier")
    st.markdown("- Ollama — local LLMs")


if "messages" not in st.session_state:
    st.session_state.messages = []

# Render the complete conversation, including saved metrics.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("result"):
            result = message["result"]
            metrics = result.get("metrics", {})
            sources = result.get("sources", [])

            with st.expander("Routing & performance"):
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Selected model",
                    metrics.get("selected_model") or "Not run",
                )
                c2.metric(
                    "Complexity",
                    metrics.get("complexity") or "Not classified",
                )

                confidence = metrics.get("routing_confidence")
                c3.metric(
                    "Classifier confidence",
                    f"{confidence * 100:.1f}%"
                    if confidence is not None else "N/A",
                )

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Total latency",
                    f"{metrics.get('end_to_end_ms', 0):.2f} ms",
                )
                c2.metric(
                    "Routing latency",
                    f"{metrics.get('routing_ms', 0):.2f} ms",
                )
                c3.metric(
                    "Generation latency",
                    f"{metrics.get('generation_ms', 0):.2f} ms",
                )

                exact_hit = metrics.get("cache_hit", False)
                semantic_hit = metrics.get("semantic_cache_hit", False)

                if exact_hit:
                    st.success("Exact cache hit")
                elif semantic_hit:
                    similarity = metrics.get("semantic_similarity")
                    suffix = (
                        f" · Similarity: {similarity:.4f}"
                        if similarity is not None else ""
                    )
                    st.success(f"Semantic cache hit{suffix}")
                else:
                    st.info("Cache miss — request processed normally")

                st.caption(
                    f"Exact lookup: {metrics.get('cache_lookup_ms', 0):.2f} ms"
                    f" · Semantic lookup: "
                    f"{metrics.get('semantic_lookup_ms', 0):.2f} ms"
                    f" · Retrieval: {metrics.get('retrieval_ms', 0):.2f} ms"
                )

            if sources:
                with st.expander("Retrieved sources"):
                    for source in sources:
                        st.markdown(
                            f"- **{source['source']}** "
                            f"(chunk {source['chunk_index']}, "
                            f"score {source['score']:.3f})"
                        )

question = st.chat_input("Ask something about your documents...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Processing your question..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=(5, 600),
                )
                response.raise_for_status()
                result = response.json()

                answer = result["answer"]
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "result": result,
                })

                st.rerun()

            except requests.Timeout:
                st.error("The request timed out. Try again.")
            except requests.HTTPError as exc:
                detail = "The API rejected the request."
                try:
                    detail = exc.response.json().get("detail", detail)
                except (ValueError, AttributeError):
                    pass
                st.error(f"API error: {detail}")
            except requests.RequestException:
                st.error(
                    "Couldn't reach FastAPI. Make sure the backend "
                    f"is running at {API_URL}."
                )
            except (KeyError, ValueError):
                st.error("The API returned an unexpected response.")
