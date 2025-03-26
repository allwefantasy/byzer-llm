import os
import sys
import asyncio
from pydantic import BaseModel
from typing import Generator, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.byzerllm.utils.client.simple_byzerllm_client import SimpleByzerLLM
from src.byzerllm.utils.client.mgenerai import MGeminiAI, MAsyncGeminiAI

# Define a Pydantic model for structured output
class SummaryResult(BaseModel):
    summary: str
    key_points: list[str]

class MetaHolder:
    def __init__(self):
        self.meta = None

async def main():
    # Check for API key
    if "MODEL_GOOGLE_TOKEN" not in os.environ:
        print("Please set GOOGLE_API_KEY environment variable")
        print("Get an API key from: https://aistudio.google.com/app/apikey")
        return

    print("=== Google Gemini Model Demo with SimpleByzerLLM ===\n")
    
    # Create the SimpleByzerLLM client
    llm = SimpleByzerLLM(default_model_name="gemini")
    
    # Deploy a model using MGeminiAI
    llm.deploy(
        model_path="gemini-1.5-pro-latest",
        pretrained_model_type="saas/gemini",
        udf_name="gemini",
        infer_params={
            "saas.api_key": os.environ["GOOGLE_API_KEY"],
            "saas.base_url": "https://generativelanguage.googleapis.com/v1beta",
            "saas.model": "gemini-1.5-pro-latest"
        }
    )
    
    print("Model deployed successfully!")
    
    # 1. Basic text completion
    print("\n=== Text Completion ===")
    prompt = "Explain how quantum computing works in simple terms."
    print(f"Prompt: {prompt}")
    
    response = llm.chat_oai(
        conversations=[{"role": "user", "content": prompt}],
        model="gemini"
    )
    
    print(f"Response: {response[0].output}")
    print(f"Token usage: {response[0].metadata.get('input_tokens_count')} input, "
          f"{response[0].metadata.get('generated_tokens_count')} output")

    # 2. Text completion with streaming
    print("\n=== Streaming Text Completion ===")
    prompt = "Count from 1 to 10, with a brief description of each number."
    print(f"Prompt: {prompt}")
    print("Response: ", end="", flush=True)
    
    for chunk, meta in llm.stream_chat_oai(
        conversations=[{"role": "user", "content": prompt}],
        model="gemini",
        delta_mode=True
    ):
        print(chunk, end="", flush=True)
    print("\n")

    # 3. Structured output using decorator
    print("\n=== Structured Output ===")
    meta_holder = MetaHolder()
    
    @llm.prompt(model="gemini", meta_holder=meta_holder)
    def summarize_text(text: str) -> SummaryResult:
        """
        Please summarize the following text and extract key points.
        
        Text: {{ text }}
        
        Respond with a concise summary and a list of key points.
        """
    
    article = """
    Artificial intelligence has made significant strides in recent years. 
    Large language models like GPT-4, Claude, and Gemini can now understand 
    and generate human-like text, translate languages, write different kinds 
    of creative content, and answer questions in an informative way. 
    However, these models also face challenges including biases in their training 
    data, potential for generating misinformation, and significant computational 
    resources required for training. Researchers are working to address these 
    limitations while exploring new applications in healthcare, education, and 
    scientific discovery.
    """
    
    result = summarize_text(text=article)
    print(f"Summary: {result.summary}")
    print("Key Points:")
    for point in result.key_points:
        print(f"- {point}")
    
    print(f"Token usage: {meta_holder.meta.input_tokens_count} input, "
          f"{meta_holder.meta.generated_tokens_count} output")

    # 4. Embeddings
    print("\n=== Text Embeddings ===")
    texts = [
        "Artificial intelligence is transforming industries.",
        "Machine learning models require large datasets.",
    ]
    
    embeddings = llm.emb(
        model="gemini",
        request={"instruction": texts[0]}
    )
    
    print(f"Text: '{texts[0]}'")
    print(f"Embedding dimension: {len(embeddings[0].output)}")
    print(f"First 5 values: {embeddings[0].output[:5]}")
    
    # 5. Async streaming example
    print("\n=== Async Streaming ===")
    prompt = "Write a short poem about artificial intelligence."
    print(f"Prompt: {prompt}")
    print("Response: ", end="", flush=True)
    
    async for chunk, meta in llm.async_stream_chat_oai(
        conversations=[{"role": "user", "content": prompt}],
        model="gemini",
        delta_mode=True
    ):
        print(chunk, end="", flush=True)
    print("\n")

    # 6. Using a generator function with streaming
    print("\n=== Generator Function ===")
    
    @llm.prompt(model="gemini")
    def generate_story() -> Generator[str, None, None]:
        """
        Write a very short story about a robot that learns to feel emotions.
        Keep it under 200 words.
        """
    
    print("Story: ", end="")
    for chunk in generate_story():
        print(chunk, end="", flush=True)
    print("\n")
    
    # Cleanup
    llm.undeploy("gemini")
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
