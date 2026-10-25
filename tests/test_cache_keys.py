from app.cache import make_cache_key

question_1 = "What is semantic caching?"
question_2 = "  WHAT IS SEMANTIC CACHING?  "
question_3 = "What   is semantic caching?"
question_4 = "What is semantic caching"

key_1 = make_cache_key(question_1)
key_2 = make_cache_key(question_2)
key_3 = make_cache_key(question_3)
key_4 = make_cache_key(question_4)

assert key_1 == key_2
assert key_1 == key_3
assert key_1 != key_4

print("Normalization test: PASSED")
print("Different question test: PASSED")
print("Generated key:", key_1)
