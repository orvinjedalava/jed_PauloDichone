import chromadb

client = chromadb.Client()


collection = client.get_or_create_collection(name="my_test_collection")
collection.add(
    ids=["1", "2", "3"],
    documents=[
        "These oranges are delicious.",
        "Some apples are sweet.",
        "Citrus fruits are great.",
    ],
)

results = collection.query(
    query_texts=["tell me about citrus fruits"],
    n_results=2,
)

print("Query Results:", results)
