"""
OpenAI SDK Usage Examples

This file demonstrates how to use the OpenAI SDK for various tasks:
- Text completion
- Chat completions
- Embeddings
- Function calling
- Error handling

Requirements:
- pip install openai
"""

import os
import time
from typing import List, Dict, Any, Optional
import json
import numpy as np

# Import OpenAI
try:
    import openai
    from openai import OpenAI
except ImportError:
    print("OpenAI SDK not found. Install with: pip install openai")
    exit(1)


def setup_client() -> OpenAI:
    """Initialize and return an OpenAI client."""
    # Load API key from environment variable (preferred) or set directly
    api_key = os.getenv("OPENAI_API_KEY","xxxx")
    
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        api_key = input("Please enter your OpenAI API key: ")
    
    # Optional: Set a custom API base for non-OpenAI deployments
    # api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    # Create a client
    client = OpenAI(api_key=api_key,base_url="http://127.0.0.1:8888/v1")
    
    return client


def basic_completion_example(client: OpenAI) -> None:
    """Basic example of text completion."""
    print("\n=== Basic Completion Example ===")
    
    try:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",  # Use a completion model
            prompt="Write a short poem about artificial intelligence:",
            max_tokens=100,
            temperature=0.7,
        )
        
        print(response.choices[0].text.strip())
        print(f"Model used: {response.model}")
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"Error in completion: {e}")


def chat_completion_example(client: OpenAI) -> None:
    """Example of chat completion API."""
    print("\n=== Chat Completion Example ===")
    
    try:
        # Example of a structured conversation
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" for more advanced capabilities
            messages=[
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a Python function to calculate Fibonacci numbers."}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        print(response.choices[0].message.content)
        print(f"Model used: {response.model}")
        print(f"Total tokens: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"Error in chat completion: {e}")


def streaming_chat_example(client: OpenAI) -> None:
    """Example of streaming chat responses."""
    print("\n=== Streaming Chat Example ===")
    
    try:
        # Stream the response
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 10, with a brief pause between each number."}
            ],
            stream=True,
            max_tokens=100
        )
        
        # Process the stream
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                time.sleep(0.05)  # Small delay to simulate typing
        print()  # Final newline
        
    except Exception as e:
        print(f"Error in streaming chat: {e}")


def function_calling_example(client: OpenAI) -> None:
    """Example of function calling with GPT models."""
    print("\n=== Function Calling Example ===")
    
    # Define available functions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g., San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    try:
        # Step 1: Send the conversation and available functions to the model
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather like in New York?"}
            ],
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        
        # Step 2: Check if the model wants to call a function
        if response_message.tool_calls:
            print("Function call requested:")
            
            # Print out the function call
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"Function: {function_name}")
            print(f"Arguments: {function_args}")
            
            # Step 3: In a real app, you'd call your actual function here
            # For this example, we'll mock a response
            function_response = {
                "location": function_args.get("location"),
                "temperature": "72°F",
                "forecast": "Sunny with some clouds",
                "humidity": "45%"
            }
            
            # Step 4: Send the function result back to the model
            second_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What's the weather like in New York?"},
                    response_message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(function_response)
                    }
                ]
            )
            
            # Get the model's interpretation of the function result
            print("\nAssistant's response after function call:")
            print(second_response.choices[0].message.content)
        else:
            print(response_message.content)
        
    except Exception as e:
        print(f"Error in function calling: {e}")


def embeddings_example(client: OpenAI) -> None:
    """Example of using embeddings."""
    print("\n=== Embeddings Example ===")
    
    try:
        # Get embeddings for two texts
        text1 = "The food was delicious and the service was excellent."
        text2 = "The meal was amazing and the staff was very attentive."
        text3 = "The weather is quite cold today."
        
        response = client.embeddings.create(
            model="emb_chat",
            input=[text1, text2, text3]
        )
        
        # Extract the embedding vectors
        embedding1 = response.data[0].embedding
        embedding2 = response.data[1].embedding
        embedding3 = response.data[2].embedding
        
        # Calculate cosine similarity between the vectors
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_1_2 = cosine_similarity(embedding1, embedding2)
        sim_1_3 = cosine_similarity(embedding1, embedding3)
        
        print(f"Embedding dimensions: {len(embedding1)}")
        print(f"Similarity between text1 and text2: {sim_1_2:.4f}")
        print(f"Similarity between text1 and text3: {sim_1_3:.4f}")
        print(f"As expected, the similar texts (1 & 2) have higher similarity than dissimilar texts (1 & 3)")
        
    except Exception as e:
        print(f"Error in embeddings: {e}")


def error_handling_example(client: OpenAI) -> None:
    """Examples of error handling with the OpenAI API."""
    print("\n=== Error Handling Example ===")
    
    try:
        # Example 1: Invalid model name
        print("Attempting to use an invalid model name...")
        response = client.chat.completions.create(
            model="non-existent-model",
            messages=[{"role": "user", "content": "Hello"}]
        )
    except openai.APIError as e:
        print(f"OpenAI API error: {e}")
    except openai.RateLimitError as e:
        print(f"Rate limit exceeded: {e}")
    except openai.APIConnectionError as e:
        print(f"Connection error: {e}")
    except openai.AuthenticationError as e:
        print(f"Authentication error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Example 2: Handling context length
    print("\nTesting context length limitations...")
    try:
        # Create a long input that might exceed context length
        long_input = "Tell me about AI. " * 1000
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": long_input}]
        )
        print("Response received successfully")
    except Exception as e:
        print(f"Error: {e}")


def cost_estimation_example() -> None:
    """Example showing how to estimate costs for API calls."""
    print("\n=== Cost Estimation Example ===")
    
    # Sample pricing (as of early 2023 - check OpenAI's website for current pricing)
    prices = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # per 1K tokens
        "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
        "text-embedding-ada-002": {"input": 0.0001}  # per 1K tokens
    }
    
    # Example usage
    example_usages = [
        {"model": "gpt-3.5-turbo", "input_tokens": 500, "output_tokens": 200},
        {"model": "gpt-4", "input_tokens": 1000, "output_tokens": 500},
        {"model": "text-embedding-ada-002", "input_tokens": 2000}
    ]
    
    total_cost = 0
    
    print("Estimated costs (USD):")
    for usage in example_usages:
        model = usage["model"]
        input_tokens = usage["input_tokens"]
        
        if model in ["gpt-3.5-turbo", "gpt-4"]:
            output_tokens = usage["output_tokens"]
            input_cost = (input_tokens / 1000) * prices[model]["input"]
            output_cost = (output_tokens / 1000) * prices[model]["output"]
            cost = input_cost + output_cost
            print(f"  {model}: {input_tokens} input tokens + {output_tokens} output tokens = ${cost:.4f}")
        else:
            cost = (input_tokens / 1000) * prices[model]["input"]
            print(f"  {model}: {input_tokens} tokens = ${cost:.4f}")
        
        total_cost += cost
    
    print(f"Total estimated cost: ${total_cost:.4f}")
    print("Note: These are approximate costs. Check OpenAI's pricing page for current rates.")


def main():
    """Run all examples."""
    print("OpenAI SDK Examples\n")
    
    client = setup_client()
    
    # Run the examples
    # basic_completion_example(client)
    # chat_completion_example(client)
    # streaming_chat_example(client)
    # function_calling_example(client)
    embeddings_example(client)
    # error_handling_example(client)
    # cost_estimation_example()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main() 