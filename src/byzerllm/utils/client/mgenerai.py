import os
import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable, Union
import google.generativeai as genai
from dataclasses import dataclass, field

# Mock OpenAI-compatible response classes
@dataclass
class MGeminiChoice:
    index: int = 0
    finish_reason: str = "stop"
    
    @dataclass
    class Message:
        content: str
        role: str = "assistant"
        reasoning_content: Optional[str] = None
        
    @dataclass
    class Delta:
        content: Optional[str] = None
        reasoning_content: Optional[str] = None
        
    message: Message = None
    delta: Delta = None

@dataclass
class MGeminiUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class MGeminiResponse:
    id: str = f"gemini-{int(time.time())}"
    choices: List[MGeminiChoice] = field(default_factory=list)
    usage: MGeminiUsage = field(default_factory=MGeminiUsage)
    
@dataclass
class MGeminiEmbeddingData:
    embedding: List[float]
    index: int = 0

@dataclass
class MGeminiEmbeddingResponse:
    data: List[MGeminiEmbeddingData]
    usage: MGeminiUsage = field(default_factory=MGeminiUsage)

# Chat Completions API
class MGeminiChatCompletions:
    def __init__(self, client):
        self.client = client
    
    def create(
        self, 
        messages: List[Dict[str, Any]], 
        model: str = "gemini-2.5-pro-exp-03-25",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        top_p: float = 1.0,
        stream: bool = False,
        stream_options: Dict[str, Any] = None,
        **kwargs
    ):
        # Convert OpenAI-style messages to Google Gemini format
        history = []
        last_message = None
        
        # Process messages to create history and extract last user message
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Handle different content formats (text vs. multimodal)
            if isinstance(content, list):  # Multimodal content
                parts = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append(item["text"])
                    elif item.get("type") == "image_url":
                        # Convert image URL to Gemini format
                        image_url = item["image_url"]["url"]
                        if image_url.startswith("data:"):  # Handle base64
                            parts.append(genai.types.Part.from_data(data=image_url))
                        else:  # URL
                            parts.append(genai.types.Part.from_uri(uri=image_url))
                    elif item.get("type") == "audio":
                        # For audio, you would need to handle accordingly
                        pass
                
                if role == "user":
                    last_message = parts
                else:
                    history.append({"role": role, "parts": parts})
            else:  # Text-only content
                if role == "user":
                    last_message = content
                else:
                    history.append({"role": role, "parts": [content]})
        
        # Create Gemini model
        gemini_model = genai.GenerativeModel(model)
        
        # Optional: Handle prefix in assistant role
        has_prefix = False
        prefix_text = ""
        
        if "extra_body" in kwargs and "prefix" in kwargs["extra_body"]:
            prefix_text = kwargs["extra_body"]["prefix"]
            has_prefix = True
        
        # Start chat session with history if needed
        if history:
            converted_history = []
            for msg in history:
                if msg["role"] == "user":
                    # Fix: Use the correct way to create chat content based on the API version
                    # For newer versions of the API
                    try:
                        converted_history.append({"role": "user", "parts": msg["parts"]})
                    except Exception:
                        # Fall back to alternative implementation if needed
                        pass
                elif msg["role"] == "assistant":
                    # Fix: Use the correct way to create chat content based on the API version
                    try:
                        converted_history.append({"role": "model", "parts": msg["parts"]})
                    except Exception:
                        # Fall back to alternative implementation if needed
                        pass
                elif msg["role"] == "system":
                    # Handle system messages - might need special treatment for Gemini
                    system_content = f"System instruction: {msg['parts'][0]}"
                    # Fix: Use the correct way to create chat content based on the API version
                    try:
                        converted_history.append({"role": "user", "parts": [system_content]})
                    except Exception:
                        # Fall back to alternative implementation if needed
                        pass
            
            chat = gemini_model.start_chat(history=converted_history)
        else:
            chat = gemini_model.start_chat(history=[])
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            stop_sequences=kwargs.get("stop", [])
        )
        
        # Handle streaming vs. non-streaming response
        if stream:
            return self._handle_stream_response(
                chat, 
                last_message, 
                generation_config, 
                has_prefix, 
                prefix_text
            )
        else:
            return self._handle_normal_response(
                chat, 
                last_message, 
                generation_config, 
                has_prefix, 
                prefix_text
            )
    
    def _handle_normal_response(
        self, 
        chat, 
        last_message, 
        generation_config, 
        has_prefix=False, 
        prefix_text=""
    ):
        try:
            if has_prefix:
                # When prefix is provided, we append it to the model's response
                response = chat.send_message(
                    last_message,
                    generation_config=generation_config
                )
                content = prefix_text + response.text
            else:
                # Normal response
                response = chat.send_message(
                    last_message,
                    generation_config=generation_config
                )
                content = response.text
            
            # Get token usage if available
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            
            if hasattr(response, "usage_metadata"):
                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
                total_tokens = getattr(response.usage_metadata, "total_token_count", 0)
            
            # Create OpenAI-compatible response
            gemini_response = MGeminiResponse(
                choices=[
                    MGeminiChoice(
                        message=MGeminiChoice.Message(
                            content=content,
                        ),
                        finish_reason="stop"
                    )
                ],
                usage=MGeminiUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )
            )
            
            return gemini_response
            
        except Exception as e:
            # Handle errors
            gemini_response = MGeminiResponse()
            gemini_response.error = str(e)
            return gemini_response
    
    def _handle_stream_response(
        self, 
        chat, 
        last_message, 
        generation_config, 
        has_prefix=False, 
        prefix_text=""
    ):
        try:
            response_stream = chat.send_message(
                last_message,
                generation_config=generation_config,
                stream=True
            )
            
            # If we have a prefix, yield it first
            if has_prefix:
                yield self._create_stream_chunk(prefix_text)
            
            # Create generator that yields OpenAI-compatible chunks
            for chunk in response_stream:
                if not hasattr(chunk, "parts") or not chunk.parts:
                    continue
                    
                # Extract text from the chunk
                text = chunk.text if hasattr(chunk, "text") else ""
                
                # Create and yield the OpenAI-compatible chunk
                yield self._create_stream_chunk(text)
            
            # Add usage information at the end if available
            if hasattr(response_stream, "usage_metadata"):
                prompt_tokens = getattr(response_stream.usage_metadata, "prompt_token_count", 0)
                completion_tokens = getattr(response_stream.usage_metadata, "candidates_token_count", 0)
                total_tokens = getattr(response_stream.usage_metadata, "total_token_count", 0)
                
                # Create and yield a final chunk with usage information
                final_chunk = MGeminiResponse(
                    choices=[MGeminiChoice(
                    delta=MGeminiChoice.Delta(
                        content=""
                    ),
                    finish_reason=None
                )],
                    usage=MGeminiUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens
                    )
                )
                yield final_chunk
                
        except Exception as e:
            # Handle errors
            error_response = MGeminiResponse()
            error_response.error = str(e)
            yield error_response
    
    def _create_stream_chunk(self, content):
        gemini_chunk = MGeminiResponse(
            choices=[
                MGeminiChoice(
                    delta=MGeminiChoice.Delta(
                        content=content
                    ),
                    finish_reason=None
                )
            ]
        )
        return gemini_chunk

# Embeddings API
class MGeminiEmbeddings:
    def __init__(self, client):
        self.client = client
        
    def create(self, input: List[str], model: str = "models/embedding-001", **kwargs):
        try:
            # Use Google's embedding model
            embedding_model = genai.get_embedding_model('models/embedding-001')
            
            # Generate embeddings for all inputs
            embeddings = []
            total_tokens = 0
            
            for i, text in enumerate(input):
                result = embedding_model.embed_content(
                    content=text,
                    task_type="retrieval_document"
                )
                
                embeddings.append(
                    MGeminiEmbeddingData(
                        embedding=result.embedding, 
                        index=i
                    )
                )
                
                # Track approximate token count (rough estimate)
                total_tokens += len(text.split()) // 3
            
            # Create OpenAI-compatible response
            return MGeminiEmbeddingResponse(
                data=embeddings,
                usage=MGeminiUsage(
                    prompt_tokens=total_tokens,
                    completion_tokens=0,
                    total_tokens=total_tokens
                )
            )
            
        except Exception as e:
            error_response = MGeminiEmbeddingResponse(data=[])
            error_response.error = str(e)
            return error_response

# Main client class that mimics OpenAI client
class MGeminiAI:
    def __init__(self, api_key=None, base_url=None, **kwargs):
        api_key = api_key
        if not api_key:
            raise ValueError("API key must be provided either as an argument or through the GOOGLE_API_KEY environment variable")
        
        # Configure genai with the API key
        genai.configure(api_key=api_key)
        
        # Create API interfaces - make structure match OpenAI's
        self.chat = self._ChatCompletionsInterface(self)
    
    class _ChatCompletionsInterface:
        def __init__(self, client):
            self.client = client
            self.completions = MGeminiChatCompletions(client)
        
    def close(self):
        # No-op for compatibility
        pass
    
    @property
    def embeddings(self):
        return MGeminiEmbeddings(self)

# Async version of the client
class MAsyncGeminiAI:
    def __init__(self, api_key=None, base_url=None, **kwargs):        
        if not api_key:
            raise ValueError("API key must be provided either as an argument or through the GOOGLE_API_KEY environment variable")
        
        # Configure genai with the API key
        genai.configure(api_key=api_key)
        
        # Create API interfaces - make structure match AsyncOpenAI
        self.chat = self._AsyncChatInterface(self)
    
    class _AsyncChatInterface:
        def __init__(self, client):
            self.client = client
            self.completions = client._AsyncChatCompletions(client)
    
    class _AsyncChatCompletions:
        def __init__(self, client):
            self.client = client
            self._sync_client = MGeminiChatCompletions(client)
        
        async def create(
            self, 
            messages: List[Dict[str, Any]], 
            model: str = "gemini-1.5-flash-latest",
            temperature: float = 0.7,
            max_tokens: int = 8192,
            top_p: float = 1.0,
            stream: bool = False,
            **kwargs
        ):
            if not stream:
                # For non-streaming requests, run synchronously in a thread
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self._sync_client.create(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stream=False,
                        **kwargs
                    )
                )
            else:
                # For streaming, we need to handle it differently
                # We'll convert the sync generator to an async generator
                sync_gen = self._sync_client.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True,
                    **kwargs
                )
                
                async def async_generator():
                    loop = asyncio.get_event_loop()
                    for chunk in sync_gen:
                        # Yield each chunk with a small delay to allow other tasks to run
                        yield chunk
                        await asyncio.sleep(0)
                
                return async_generator()
    
    @property
    def embeddings(self):
        return self._AsyncEmbeddings(self)
            
    class _AsyncEmbeddings:
        def __init__(self, client):
            self.client = client
            self._sync_client = MGeminiEmbeddings(client)
        
        async def create(self, input: List[str], model: str = "models/embedding-001", **kwargs):
            # Run embedding creation in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._sync_client.create(input=input, model=model, **kwargs)
            )
    
    async def close(self):
        # No-op for compatibility
        pass
