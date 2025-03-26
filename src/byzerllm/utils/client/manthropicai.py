import os
import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable, Union
import anthropic
from dataclasses import dataclass, field

# Mock OpenAI-compatible response classes
@dataclass
class MAnthropicChoice:
    index: int = 0
    finish_reason: str = "stop"
    
    @dataclass
    class Message:
        content: str
        role: str = "assistant"
        
    @dataclass
    class Delta:
        content: Optional[str] = None
        
    message: Message = None
    delta: Delta = None

@dataclass
class MAnthropicUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class MAnthropicResponse:
    id: str = f"claude-{int(time.time())}"
    choices: List[MAnthropicChoice] = field(default_factory=list)
    usage: MAnthropicUsage = field(default_factory=MAnthropicUsage)
    
# Chat Completions API
class MAnthropicChatCompletions:
    def __init__(self, client):
        self.client = client
    
    def create(
        self, 
        messages: List[Dict[str, Any]], 
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs
    ):
        # Convert OpenAI-style messages to Anthropic format
        system_message = None
        anthropic_messages = []
        
        # Process messages to create proper format for Anthropic API
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_message = content
            elif role == "user":
                if isinstance(content, list):  # Multimodal content
                    parts = []
                    for item in content:
                        if item.get("type") == "text":
                            parts.append({"type": "text", "text": item["text"]})
                        elif item.get("type") == "image_url":
                            # Convert image URL to Anthropic format
                            image_url = item["image_url"]["url"]
                            media_type = "image/jpeg"  # Default
                            
                            # Try to determine media type from URL
                            if image_url.endswith(".png"):
                                media_type = "image/png"
                            elif image_url.endswith(".jpg") or image_url.endswith(".jpeg"):
                                media_type = "image/jpeg"
                            
                            if image_url.startswith("data:"):
                                # Handle base64 images
                                parts.append({
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": media_type, "data": image_url.split(",")[1]}
                                })
                            else:
                                # Handle URL images
                                parts.append({
                                    "type": "image",
                                    "source": {"type": "url", "url": image_url}
                                })
                    
                    anthropic_messages.append({"role": "user", "content": parts})
                else:  # Text-only content
                    anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                if isinstance(content, list):
                    # Handle complex content if needed
                    anthropic_messages.append({"role": "assistant", "content": content})
                else:
                    anthropic_messages.append({"role": "assistant", "content": content})
        
        # Optional: Handle prefix in assistant role
        has_prefix = False
        prefix_text = ""
        
        if "extra_body" in kwargs and "prefix" in kwargs["extra_body"]:
            prefix_text = kwargs["extra_body"]["prefix"]
            has_prefix = True
        
        # Handle streaming vs. non-streaming response
        if stream:
            return self._handle_stream_response(
                anthropic_messages, 
                model, 
                system_message, 
                temperature, 
                max_tokens, 
                top_p, 
                has_prefix, 
                prefix_text,
                kwargs.get("stop", [])
            )
        else:
            return self._handle_normal_response(
                anthropic_messages, 
                model, 
                system_message, 
                temperature, 
                max_tokens, 
                top_p, 
                has_prefix, 
                prefix_text,
                kwargs.get("stop", [])
            )
    
    def _handle_normal_response(
        self, 
        messages, 
        model, 
        system_message, 
        temperature, 
        max_tokens, 
        top_p, 
        has_prefix=False, 
        prefix_text="",
        stop_sequences=[]
    ):
        try:
            # Create the message for Anthropic API
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            
            if system_message:
                kwargs["system"] = system_message
                
            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences
                
            # Make the API call
            response = self.client.client.messages.create(**kwargs)
            
            content = response.content[0].text
            if has_prefix:
                content = prefix_text + content
            
            # Create OpenAI-compatible response
            anthropic_response = MAnthropicResponse(
                id=response.id,
                choices=[
                    MAnthropicChoice(
                        message=MAnthropicChoice.Message(
                            content=content,
                            role="assistant"
                        ),
                        finish_reason=response.stop_reason if hasattr(response, "stop_reason") else "stop"
                    )
                ],
                usage=MAnthropicUsage(
                    prompt_tokens=response.usage.input_tokens if hasattr(response, "usage") else 0,
                    completion_tokens=response.usage.output_tokens if hasattr(response, "usage") else 0,
                    total_tokens=(
                        response.usage.input_tokens + response.usage.output_tokens
                        if hasattr(response, "usage")
                        else 0
                    )
                )
            )
            
            return anthropic_response
            
        except Exception as e:
            # Handle errors
            anthropic_response = MAnthropicResponse()
            anthropic_response.error = str(e)
            return anthropic_response
    
    def _handle_stream_response(
        self, 
        messages, 
        model, 
        system_message, 
        temperature, 
        max_tokens, 
        top_p, 
        has_prefix=False, 
        prefix_text="",
        stop_sequences=[]
    ):
        try:
            # Create the message for Anthropic API with streaming
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p                
            }
            
            if system_message:
                kwargs["system"] = system_message
                
            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences
                
            # Support for thinking parameter if available in kwargs
            if "extra_body" in kwargs and "thinking" in kwargs["extra_body"]:
                kwargs["thinking"] = kwargs["extra_body"]["thinking"]
                
            # Create a stream with context manager to properly handle resources
            with self.client.client.messages.stream(**kwargs) as stream:
                # If we have a prefix, yield it first
                if has_prefix:
                    yield self._create_stream_chunk(prefix_text)
                
                # Track for usage information 
                input_tokens = 0
                output_tokens = 0
                
                # Process each chunk/event in the stream
                for event in stream:
                    if event.type == "content_block_start":
                        # Handle start of a content block
                        pass
                    elif event.type == "content_block_delta":
                        if hasattr(event, "delta") and hasattr(event.delta, "text"):
                            text = event.delta.text
                            # Create and yield the OpenAI-compatible chunk
                            yield self._create_stream_chunk(text)
                    elif event.type == "content_block_stop":
                        # Handle end of a content block
                        pass
                    elif event.type == "message_start":
                        # Handle start of message
                        pass
                    elif event.type == "message_delta":
                        # Handle message delta 
                        pass
                    elif event.type == "message_stop":
                        # Handle end of message
                        pass
                    elif event.type == "thinking":
                        # Handle thinking output if needed
                        # For now, we don't expose this in the OpenAI-compatible interface
                        pass
                    elif event.type == "text":
                        # Direct text output (common in newer SDK versions)
                        yield self._create_stream_chunk(event.text)
                    
                    # Track token usage if available
                    if hasattr(event, "usage") and event.usage:
                        if hasattr(event.usage, "input_tokens"):
                            input_tokens = event.usage.input_tokens
                        if hasattr(event.usage, "output_tokens"):
                            output_tokens = event.usage.output_tokens
                
                # Try to get final message for usage information
                try:
                    final_message = stream.get_final_message()
                    if hasattr(final_message, "usage"):
                        input_tokens = final_message.usage.input_tokens
                        output_tokens = final_message.usage.output_tokens
                except:
                    # If get_final_message isn't supported or fails
                    pass
                
                # Add usage information at the end
                final_chunk = MAnthropicResponse(
                    choices=[MAnthropicChoice(
                        delta=MAnthropicChoice.Delta(
                            content=""
                        ),
                        finish_reason="stop"
                    )],
                    usage=MAnthropicUsage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens
                    )
                )
                yield final_chunk
                
        except Exception as e:
            # Handle errors
            error_response = MAnthropicResponse()
            error_response.error = str(e)
            yield error_response
    
    def _create_stream_chunk(self, content):
        anthropic_chunk = MAnthropicResponse(
            choices=[
                MAnthropicChoice(
                    delta=MAnthropicChoice.Delta(
                        content=content
                    ),
                    finish_reason=None
                )
            ]
        )
        return anthropic_chunk

# Main client class that mimics OpenAI client
class MAnthropicAI:
    def __init__(self, api_key=None, base_url=None, **kwargs):
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("API key must be provided either as an argument or through the ANTHROPIC_API_KEY environment variable")
        
        # Initialize the Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Create API interfaces - make structure match OpenAI's
        self.chat = self._ChatCompletionsInterface(self)
    
    class _ChatCompletionsInterface:
        def __init__(self, client):
            self.client = client
            self.completions = MAnthropicChatCompletions(client)
        
    def close(self):
        # No-op for compatibility
        pass

# Async version of the client
class MAsyncAnthropicAI:
    def __init__(self, api_key=None, base_url=None, **kwargs):        
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("API key must be provided either as an argument or through the ANTHROPIC_API_KEY environment variable")
        
        # Initialize the Anthropic async client
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        
        # Create API interfaces - make structure match AsyncOpenAI
        self.chat = self._AsyncChatInterface(self)
    
    class _AsyncChatInterface:
        def __init__(self, client):
            self.client = client
            self.completions = client._AsyncChatCompletions(client)
    
    class _AsyncChatCompletions:
        def __init__(self, client):
            self.client = client
        
        async def create(
            self, 
            messages: List[Dict[str, Any]], 
            model: str = "claude-3-opus-20240229",
            temperature: float = 0.7,
            max_tokens: int = 4096,
            top_p: float = 1.0,
            stream: bool = False,
            **kwargs
        ):
            # Convert OpenAI-style messages to Anthropic format
            system_message = None
            anthropic_messages = []
            
            # Process messages to create proper format for Anthropic API
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    system_message = content
                elif role == "user":
                    if isinstance(content, list):  # Multimodal content
                        parts = []
                        for item in content:
                            if item.get("type") == "text":
                                parts.append({"type": "text", "text": item["text"]})
                            elif item.get("type") == "image_url":
                                # Convert image URL to Anthropic format
                                image_url = item["image_url"]["url"]
                                media_type = "image/jpeg"  # Default
                                
                                # Try to determine media type from URL
                                if image_url.endswith(".png"):
                                    media_type = "image/png"
                                elif image_url.endswith(".jpg") or image_url.endswith(".jpeg"):
                                    media_type = "image/jpeg"
                                
                                if image_url.startswith("data:"):
                                    # Handle base64 images
                                    parts.append({
                                        "type": "image",
                                        "source": {"type": "base64", "media_type": media_type, "data": image_url.split(",")[1]}
                                    })
                                else:
                                    # Handle URL images
                                    parts.append({
                                        "type": "image",
                                        "source": {"type": "url", "url": image_url}
                                    })
                        
                        anthropic_messages.append({"role": "user", "content": parts})
                    else:  # Text-only content
                        anthropic_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    if isinstance(content, list):
                        # Handle complex content if needed
                        anthropic_messages.append({"role": "assistant", "content": content})
                    else:
                        anthropic_messages.append({"role": "assistant", "content": content})
            
            # Optional: Handle prefix in assistant role
            has_prefix = False
            prefix_text = ""
            
            if "extra_body" in kwargs and "prefix" in kwargs["extra_body"]:
                prefix_text = kwargs["extra_body"]["prefix"]
                has_prefix = True
            
            # Create the message for Anthropic API
            api_kwargs = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            
            if system_message:
                api_kwargs["system"] = system_message
                
            if "stop" in kwargs and kwargs["stop"]:
                api_kwargs["stop_sequences"] = kwargs["stop"]
                
            try:
                if not stream:
                    # Handle normal response
                    response = await self.client.client.messages.create(**api_kwargs)
                    
                    content = response.content[0].text
                    if has_prefix:
                        content = prefix_text + content
                    
                    # Create OpenAI-compatible response
                    anthropic_response = MAnthropicResponse(
                        id=response.id,
                        choices=[
                            MAnthropicChoice(
                                message=MAnthropicChoice.Message(
                                    content=content,
                                    role="assistant"
                                ),
                                finish_reason=response.stop_reason if hasattr(response, "stop_reason") else "stop"
                            )
                        ],
                        usage=MAnthropicUsage(
                            prompt_tokens=response.usage.input_tokens if hasattr(response, "usage") else 0,
                            completion_tokens=response.usage.output_tokens if hasattr(response, "usage") else 0,
                            total_tokens=(
                                response.usage.input_tokens + response.usage.output_tokens
                                if hasattr(response, "usage")
                                else 0
                            )
                        )
                    )
                    
                    return anthropic_response
                else:
                    # Handle streaming
                    api_kwargs["stream"] = True
                    
                    # Support for thinking parameter if available in kwargs
                    if "extra_body" in kwargs and "thinking" in kwargs["extra_body"]:
                        api_kwargs["thinking"] = kwargs["extra_body"]["thinking"]
                    
                    # For streaming, return an async generator
                    async def async_generator():
                        try:
                            # If we have a prefix, yield it first
                            if has_prefix:
                                yield MAnthropicResponse(
                                    choices=[
                                        MAnthropicChoice(
                                            delta=MAnthropicChoice.Delta(
                                                content=prefix_text
                                            ),
                                            finish_reason=None
                                        )
                                    ]
                                )
                            
                            # Track for usage information 
                            input_tokens = 0
                            output_tokens = 0
                            
                            # Use async context manager for proper resource handling
                            async with await self.client.client.messages.stream(**api_kwargs) as stream:
                                # Process each chunk
                                async for event in stream:
                                    if event.type == "content_block_delta":
                                        if hasattr(event, "delta") and hasattr(event.delta, "text"):
                                            text = event.delta.text
                                            # Create and yield the OpenAI-compatible chunk
                                            yield MAnthropicResponse(
                                                choices=[
                                                    MAnthropicChoice(
                                                        delta=MAnthropicChoice.Delta(
                                                            content=text
                                                        ),
                                                        finish_reason=None
                                                    )
                                                ]
                                            )
                                    elif event.type == "text":
                                        # Direct text output (common in newer SDK versions)
                                        yield MAnthropicResponse(
                                            choices=[
                                                MAnthropicChoice(
                                                    delta=MAnthropicChoice.Delta(
                                                        content=event.text
                                                    ),
                                                    finish_reason=None
                                                )
                                            ]
                                        )
                                    
                                    # Track token usage if available
                                    if hasattr(event, "usage") and event.usage:
                                        if hasattr(event.usage, "input_tokens"):
                                            input_tokens = event.usage.input_tokens
                                        if hasattr(event.usage, "output_tokens"):
                                            output_tokens = event.usage.output_tokens
                                
                                # Try to get final message for usage information
                                try:
                                    final_message = await stream.get_final_message()
                                    if hasattr(final_message, "usage"):
                                        input_tokens = final_message.usage.input_tokens
                                        output_tokens = final_message.usage.output_tokens
                                except:
                                    # If get_final_message isn't supported or fails
                                    pass
                            
                            # Add usage information at the end
                            final_chunk = MAnthropicResponse(
                                choices=[MAnthropicChoice(
                                    delta=MAnthropicChoice.Delta(
                                        content=""
                                    ),
                                    finish_reason="stop"
                                )],
                                usage=MAnthropicUsage(
                                    prompt_tokens=input_tokens,
                                    completion_tokens=output_tokens,
                                    total_tokens=input_tokens + output_tokens
                                )
                            )
                            yield final_chunk
                                
                        except Exception as e:
                            # Handle errors
                            error_response = MAnthropicResponse()
                            error_response.error = str(e)
                            yield error_response
                    
                    return async_generator()
                    
            except Exception as e:
                # Handle errors
                anthropic_response = MAnthropicResponse()
                anthropic_response.error = str(e)
                return anthropic_response
    
    async def close(self):
        # Close the client if necessary
        if hasattr(self.client, "close") and callable(self.client.close):
            await self.client.close()
