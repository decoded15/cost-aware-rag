
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"


def generate_answer(question: str, context: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a concise document question-answering assistant. "
                        "Answer the question directly in at most 3 sentences. "
                        "Use only facts explicitly stated in the supplied context. "
                        "Do not explain your reasoning or repeat the question. "
                        "If the context does not contain the answer, say so. "
                        "Include the relevant source label in your answer."
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
            "options" : {
                "num_predict" : 256,
                "temperature" : 0.1,
            },
        },
        timeout=(10, 600),
    )

    response.raise_for_status()
    return response.json()["message"]["content"].strip()