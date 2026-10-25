from app.model_router import select_model

questions = [
    "What is caching?",
    "Explain recursion in simple terms.",
    "Compare two RAG architectures and evaluate their trade-offs.",
    "Design a distributed caching system with fault tolerance.",
    "Write a Python function to reverse a string.",
]

for question in questions:
    result = select_model(question)
    print(f"\nQuestion: {question}")
    print(f"Prediction: {result}")