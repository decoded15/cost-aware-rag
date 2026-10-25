from transformers import pipeline

MODEL_ID = "tripathyShaswata/QueryComplexityRouter"

_classifier = pipeline(
    "text-classification",
    model=MODEL_ID,
    device=-1,  
)

SIMPLE_MODEL = "qwen2.5:3b"
COMPLEX_MODEL = "llama3.2:3b"

LABEL_TO_MODEL = {
    "no_llm": SIMPLE_MODEL,
    "small_llm": SIMPLE_MODEL,
    "large_llm": COMPLEX_MODEL,
}


def select_model(question: str) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    prediction = _classifier(question[:4000], truncation=True)[0]
    label = prediction["label"]
    confidence = float(prediction["score"])

    model = LABEL_TO_MODEL[label]

    return {
        "model": model,
        "complexity": label,
        "confidence": confidence,
    }