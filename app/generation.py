
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:4b"


def generate_answer(question: str, context: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using only the "
                        "provided context. If the context doesn't "
                        "contain the answer, say you don't have "
                        "enough information. Cite sources using the "
                        "source labels included in the context."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()
    return response.json()["message"]["content"].strip()
