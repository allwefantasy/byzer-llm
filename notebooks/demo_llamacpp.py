import llama_cpp

llm = llama_cpp.Llama(model_path="/Users/allwefantasy/.auto-coder/storage/models/bge-m3-GGUF/bge-m3-Q2_K.gguf", embedding=True)

embeddings = llm.create_embedding("Hello, world!")

# or create multiple embeddings at once

embeddings = llm.create_embedding(["Hello, world!", "Goodbye, world!"])
# print(embeddings)
print(embeddings["data"][0]["embedding"])
print(embeddings["usage"]["prompt_tokens"])
print(embeddings["usage"]["total_tokens"])