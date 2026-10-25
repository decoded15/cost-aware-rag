from app.semantic_embeddings import (
    load_semantic_embedding_model,
    embed_semantic_question,
)

model = load_semantic_embedding_model()

try:
    question = "How can I reduce repeated LLM calls?"
    embedding = embed_semantic_question(question, model)

    assert len(embedding) == 384
    assert all(isinstance(value, float) for value in embedding)

    print("Semantic embedding test: PASS")
    print(f"Model: BAAI/bge-small-en-v1.5")
    print(f"Vector dimensions: {len(embedding)}")
finally:
    del model
