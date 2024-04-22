import os
import pytest
from byzerllm.auto.backend_llama_cpp import LlamaCppBackend

@pytest.fixture(scope="module")
def llama_cpp_backend():
    model_path = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/7B/llama-model.gguf")
    backend = LlamaCppBackend(model_path=model_path)
    backend.load()
    return backend

def test_generate_single_prompt(llama_cpp_backend):
    prompt = "Hello, my name is"
    output = llama_cpp_backend.generate(prompt, max_tokens=10)
    assert isinstance(output, str)
    assert output.startswith(prompt)

def test_generate_multiple_prompts(llama_cpp_backend):
    prompts = ["Hello, my name is", "The weather today is"]
    outputs = llama_cpp_backend.generate(prompts, max_tokens=10)
    assert isinstance(outputs, list)
    assert len(outputs) == len(prompts)
    for i, output in enumerate(outputs):
        assert isinstance(output, str)
        assert output.startswith(prompts[i])

def test_embed_single_text(llama_cpp_backend):
    text = "This is a test sentence."
    embedding = llama_cpp_backend.embed(text)
    assert isinstance(embedding, list)
    assert len(embedding) == 1
    assert isinstance(embedding[0], list)
    assert len(embedding[0]) > 0

def test_embed_multiple_texts(llama_cpp_backend):
    texts = ["This is a test sentence.", "Here is another test."]
    embeddings = llama_cpp_backend.embed(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    for embedding in embeddings:
        assert isinstance(embedding, list)
        assert len(embedding) > 0