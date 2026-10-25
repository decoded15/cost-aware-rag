from app.generation import generate_answer

question = "What is semantic caching?"

context = """
[Source: data/sample.txt]
Semantic caching can reuse answers to sufficiently similar questions.
Cache validity and similarity thresholds must be handled carefully
to avoid returning incorrect answers.
"""

answer = generate_answer(question, context)

print("\nQuestion:", question)
print("\nGenerated answer:\n", answer)
