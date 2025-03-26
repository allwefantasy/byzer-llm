import os
import sys
import asyncio
from typing import List
from pathlib import Path
import traceback
from byzerllm.utils.client.manthropicai import MAnthropicAI, MAsyncAnthropicAI

api_key = os.environ.get("MODEL_ANTHROPIC_TOKEN")

async def demonstrate_async_usage():
    """Demonstrate asynchronous usage of MAsyncAnthropicAI"""
    print("\n=== Asynchronous Claude API ===")

    model = "claude-3-opus-20240229"
    
    # Initialize the async client
    try:
        client = MAsyncAnthropicAI(api_key=api_key)
        
        # Simple chat completion
        messages = [
            {"role": "user", "content": "What are the three laws of robotics?"}
        ]
        
        print("Question: What are the three laws of robotics?")
        response = await client.chat.completions.create(
            messages=messages,
            model=model, 
            temperature=0.7
        )
        
        print(f"Response: {response.choices[0].message.content}\n")
        print(f"Token usage: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output tokens")
        
        # Streaming response example
        messages = [
            {"role": "user", "content": "Write a short haiku about artificial intelligence."}
        ]
        
        print("\nStreaming a haiku about AI...")
        stream = await client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.7,
            stream=True
        )
        
        print("Response: ", end="")
        full_response = ""
        async for chunk in stream:
            if hasattr(chunk.choices[0], "delta") and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print("\n")
        
        await client.close()
        
    except Exception as e:
        print(f"Async demonstration failed: {e}")
        traceback.print_exc()

def demonstrate_sync_usage():
    """Demonstrate synchronous usage of MAnthropicAI"""
    print("=== Synchronous Claude API ===")

    model = "claude-3-opus-20240229"
    
    try:
        # Initialize the client
        client = MAnthropicAI(api_key=api_key)
        
        # Basic chat completion - simplest case first
        messages = [
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ]
        
        print("Question: Explain quantum computing in simple terms.")
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.7
        )
        
        print(f"Response: {response.choices[0].message.content}\n")
        print(f"Token usage: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output tokens")
        
        # Simple streaming example 
        print("\nStreaming simple response...")
        print("Question: Count from 1 to 5")
        print("Response: ", end="")
        
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            model=model,
            temperature=0.7,
            stream=True
        )
        
        for chunk in stream:
            if hasattr(chunk.choices[0], "delta") and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")

        # Try complex conversation with history - just with user and assistant roles
        conversation = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
            {"role": "user", "content": "Can you explain what quantum entanglement is?"}
        ]
        
        print("\nSimple conversation with history:")
        print("Question: Can you explain what quantum entanglement is?")
        response = client.chat.completions.create(
            messages=conversation,
            model=model,
            temperature=0.7
        )
        
        print(f"Response: {response.choices[0].message.content}\n")
        
        client.close()
        
    except Exception as e:
        print(f"Sync demonstration failed: {e}")
        traceback.print_exc()

def main():
    if not api_key:
        print("Please set MODEL_ANTHROPIC_TOKEN environment variable")
        print("Get an API key from: https://console.anthropic.com/")
        return
                
    print("=== Claude Direct API Demo ===")
    print("This demo shows how to use MAnthropicAI and MAsyncAnthropicAI wrappers directly.\n")        
    
    try:
        demonstrate_sync_usage()
        # Uncomment to run async demo
        asyncio.run(demonstrate_async_usage())
    except Exception as e:
        print(f"Demo execution failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
