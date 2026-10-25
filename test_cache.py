import redis

cache = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
)

print("Server response:", cache.ping())

cache.set("project:greeting", "Hello from our RAG gateway!")

message = cache.get("project:greeting")
print("Cached value:", message)

cache.delete("project:greeting")

cache.close()
