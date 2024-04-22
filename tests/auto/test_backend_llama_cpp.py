import os
import pytest
import asyncio
from byzerllm.auto.backend_llama_cpp import LlamaCppBackend
import byzerllm

@pytest.fixture
def llama_cpp_backend():        
    byzerllm.connect_cluster()    
    backend = LlamaCppBackend(model_path="/Users/allwefantasy/Downloads/Meta-Llama-3-8B.Q2_K.gguf")    
    return backend

def test_generate_single_prompt(llama_cpp_backend):    
    prompt = "Hello, my name is"
    output = asyncio.run(llama_cpp_backend.generate(None,prompt, max_tokens=10))
    print(output)
    
