---
description: LLM prompt decorator for structured interaction.
globs: ["src/utils/llm_prompting.py", "*/llm_clients/*.py"]
alwaysApply: false
---

# LLM Prompt Decorator Pattern

## 简要说明
提供一个 Python 装饰器 (`@prompt`)，它将带有类型注解和 Jinja2 模板化文档字符串的函数转换为结构化的 LLM (大语言模型) API 调用。该模式简化了提示词构建、API 请求发送（同步和流式）以及结果解析（字符串、Pydantic 模型或生成器）。适用于需要与 LLM 进行结构化交互并期望特定格式输出的场景。

## 典型用法

```python
import inspect
import functools
import pydantic
from pydantic import BaseModel
from typing import Dict, Any, Optional, Generator, List, Tuple
from collections.abc import Generator as GeneratorABC
import time
from loguru import logger

# Assume this utility function exists for formatting Jinja2 templates from docstrings
def format_prompt_jinja2(func, **kwargs) -> str:
    """
    Renders the docstring of a function using Jinja2 templating.
    """
    import jinja2
    docstring = inspect.getdoc(func)
    if not docstring:
        raise ValueError(f"Function {func.__name__} must have a docstring.")
    
    # Basic Jinja2 rendering for demonstration
    env = jinja2.Environment()
    template = env.from_string(docstring)
    return template.render(**kwargs)

# Simplified LLM Response Structures (Mimicking original types)
class LLMResponse:
    def __init__(self, output: Any, metadata: Dict[str, Any], input: Any):
        self.output = output
        self.metadata = metadata
        self.input = input
        self.value = output # For Pydantic model responses
        self.response = output # Raw response text

class SingleOutputMeta(BaseModel):
    input_tokens_count: int = 0
    generated_tokens_count: int = 0
    reasoning_content: Optional[str] = None
    finish_reason: Optional[str] = None

class MetaHolder:
    """A simple class to hold metadata passed from the LLM call."""
    def __init__(self):
        self.meta: Optional[Union[Dict[str, Any], SingleOutputMeta]] = None


# Simplified LLM Client with the prompt decorator
class SimpleLLMClient:
    """
    A simplified LLM client demonstrating the prompt decorator pattern.
    In a real application, this would wrap actual API calls (e.g., OpenAI).
    """
    def __init__(self, default_model: str = "mock-model"):
        self.default_model = default_model
        logger.info(f"SimpleLLMClient initialized with default model: {default_model}")

    # --- Placeholder LLM Interaction Methods ---
    # In a real implementation, these would call the actual LLM API
    def chat_oai(self, model: str, conversations: List[Dict[str, str]], response_class: Optional[Any] = None, **kwargs) -> List[LLMResponse]:
        """Placeholder for synchronous chat completion API call."""
        logger.info(f"Simulating chat_oai call to model '{model}' with options: {kwargs}")
        prompt = conversations[-1]["content"]
        # Simulate response based on response_class
        if response_class and issubclass(response_class, BaseModel):
            try:
                # Simulate structured JSON output and parse it
                simulated_json_output = '{"name": "Example Name", "value": 123}'
                parsed_output = response_class.model_validate_json(simulated_json_output)
                output = parsed_output
                response_text = simulated_json_output
            except Exception as e:
                 logger.error(f"Failed to simulate Pydantic parsing: {e}")
                 output = None # Indicate failure
                 response_text = f"Error: Could not generate valid {response_class.__name__}"

        else:
            # Simulate simple string output
            output = f"Response for: {prompt[:50]}..."
            response_text = output

        metadata = {
            "model": model,
            "request_id": f"mock_req_{time.time()}",
            "input_tokens_count": len(prompt.split()),
            "generated_tokens_count": len(response_text.split()),
            "time_cost": 0.1,
            "finish_reason": "stop",
        }
        resp = LLMResponse(output=output, metadata=metadata, input=prompt)
        resp.response = response_text # Store raw simulated text
        return [resp]

    def stream_chat_oai(self, model: str, conversations: List[Dict[str, str]], delta_mode: bool = True, **kwargs) -> Generator[Tuple[str, SingleOutputMeta], None, None]:
        """Placeholder for streaming chat completion API call."""
        logger.info(f"Simulating stream_chat_oai call to model '{model}' with options: {kwargs}")
        prompt = conversations[-1]["content"]
        simulated_response = f"Streaming response for: {prompt[:50]}..."
        words = simulated_response.split()
        
        current_output = ""
        total_generated_tokens = 0
        meta = SingleOutputMeta(input_tokens_count=len(prompt.split()), finish_reason=None)

        for i, word in enumerate(words):
            chunk = word + " "
            total_generated_tokens += 1
            meta.generated_tokens_count = total_generated_tokens
            
            if delta_mode:
                yield (chunk, meta)
            else:
                current_output += chunk
                yield (current_output.strip(), meta)
            time.sleep(0.05) # Simulate network latency

        meta.finish_reason = "stop"
        if delta_mode:
             yield ("", meta) # Final meta update
        else:
             yield (current_output.strip(), meta)


    # --- The Prompt Decorator ---
    def prompt(
        self,
        model: Optional[str] = None,
        render: Optional[str] = "jinja2", # Specify the rendering engine
        check_result: bool = False,      # Option to log warning if Pydantic parsing fails
        options: Dict[str, Any] = {},    # Extra options for the LLM call (e.g., temperature)
        return_origin_response: bool = False, # Return raw LLMResponse object(s)
        marker: Optional[str] = None,         # Optional marker to append to the prompt
        assistant_prefix: Optional[str] = None, # Optional prefix for assistant's response (useful for few-shot)
        meta_holder: Optional[MetaHolder] = None, # Optional object to store metadata
    ):
        """
        Decorator to turn a Python function into an LLM prompt call.

        Uses the function's docstring (rendered with Jinja2) as the prompt,
        and function arguments as template variables. Determines call type (sync, stream)
        and response parsing based on the function's return type annotation.
        """
        if model is None:
            # Allow overriding model via options, otherwise use client default
            final_model = options.pop("model", self.default_model)
        else:
            final_model = model

        # Helper to check if a type annotation is for a Generator[str, None, None]
        def is_instance_of_str_generator(v):
            if inspect.isgeneratorfunction(v) or isinstance(v, GeneratorABC):
                 return True # Simple check for demo
            try:
                origin = getattr(v, "__origin__", None)
                args = getattr(v, "__args__", [])
                # Check specifically for Generator[str, None, None] or similar
                return origin in (Generator, GeneratorABC) and args and args[0] is str
            except Exception:
                 return False

        def _impl(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                signature = inspect.signature(func)
                arguments = signature.bind(*args, **kwargs)
                arguments.apply_defaults()
                input_dict = {}
                for param in signature.parameters:
                    input_dict[param] = arguments.arguments[param]

                # Allow the decorated function to modify/add context before templating
                instance = None
                if "self" in input_dict:
                    instance = input_dict.pop("self")
                    new_input_dic = func(instance, **input_dict)
                else:
                    new_input_dic = func(**input_dict)

                if new_input_dic and not isinstance(new_input_dic, dict):
                    raise TypeError(
                        f"Return value of {func.__name__} (before LLM call) should be a dict or None"
                    )
                if new_input_dic:
                    input_dict.update(new_input_dic)

                # Render the prompt using the specified engine (only Jinja2 shown)
                if render == "jinja2":
                    prompt_str = format_prompt_jinja2(func, **input_dict)
                else:
                    raise ValueError(f"Unsupported render engine: {render}")

                if marker:
                    prompt_str = f"{prompt_str}\n\n{marker}"

                # Prepare conversation history for the API call
                conversations = [{"role": "user", "content": prompt_str}]
                if assistant_prefix:
                    conversations.append({"role": "assistant", "content": assistant_prefix})

                # Determine call type and response handling based on return annotation
                return_annotation = signature.return_annotation

                # --- Streaming Response ---
                if is_instance_of_str_generator(return_annotation):
                    temp_options = {**{"delta_mode": True}, **options} # Default to delta for generators
                    
                    stream = self.stream_chat_oai(
                        conversations=conversations,
                        model=final_model,
                        **temp_options,
                    )

                    # Wrapper generator to potentially capture metadata
                    def generator_wrapper():
                        final_meta = None
                        for item, meta in stream:
                            if meta:
                                final_meta = meta # Keep track of the latest metadata
                            yield item
                        # After iteration, store the final metadata if holder exists
                        if meta_holder and final_meta:
                            meta_holder.meta = final_meta

                    if return_origin_response:
                        # Note: Returning the raw stream might be complex if metadata capture is needed post-iteration
                        logger.warning("return_origin_response=True for streams might not capture final metadata correctly.")
                        return stream

                    return generator_wrapper()

                # --- Pydantic Model Response ---
                elif inspect.isclass(return_annotation) and issubclass(return_annotation, pydantic.BaseModel):
                    response_class = return_annotation
                    llm_responses = self.chat_oai(
                        model=final_model,
                        conversations=conversations,
                        response_class=response_class,
                        # Pass original function args if needed by chat_oai for context/tool use
                        # impl_func_params=input_dict,
                        **options,
                    )
                    response = llm_responses[0] # Assuming single response for simplicity

                    if meta_holder and response.metadata:
                        meta_holder.meta = response.metadata

                    if return_origin_response:
                        return llm_responses

                    # Check if Pydantic parsing was successful (value is not None)
                    if response.value is None and check_result:
                        logger.warning(
                            f"LLM call for {func.__name__} returned None or failed Pydantic parsing. "
                            f"Metadata: {response.metadata}. Raw Response: '{response.response}'"
                        )
                    return response.value # Return the parsed Pydantic model instance

                # --- Simple String Response ---
                elif return_annotation is str:
                    llm_responses = self.chat_oai(
                        model=final_model,
                        conversations=conversations,
                        **options,
                    )
                    response = llm_responses[0]

                    if meta_holder and response.metadata:
                        meta_holder.meta = response.metadata

                    if return_origin_response:
                        return llm_responses
                    return response.output # Return the string output

                # --- Unsupported Return Type ---
                else:
                    raise TypeError(
                        f"Unsupported return type annotation '{return_annotation}' for function {func.__name__}. "
                        "Supported types: str, pydantic.BaseModel, Generator[str, None, None]."
                    )

            return wrapper
        return _impl

# --- Example Usage ---

# Example Pydantic Model
class ExtractionResult(BaseModel):
    name: str
    value: int

# Instantiate the client
llm_client = SimpleLLMClient(default_model="gpt-3.5-turbo-mock")

# Example 1: Simple string generation
@llm_client.prompt()
def generate_greeting(name: str) -> str:
    """
    Generate a friendly greeting for {{ name }}.
    """
    # This function body is executed *before* templating.
    # It can be used to prepare data for the template.
    # Returning a dict here updates the template context.
    logger.info(f"Preparing greeting for {name}")
    return {"title": "Dr."} # Adds 'title' to the template context

# Example 2: Structured data extraction (Pydantic)
@llm_client.prompt(model="gpt-4-mock", options={"temperature": 0.1}, check_result=True)
def extract_info(text: str) -> ExtractionResult:
    """
    Extract the name and value from the following text:
    {{ text }}
    Format the output as a JSON object with keys "name" and "value".
    """
    pass # No pre-processing needed

# Example 3: Streaming generation
@llm_client.prompt(options={"max_tokens": 50})
def stream_story(topic: str) -> Generator[str, None, None]:
    """
    Write a short story about {{ topic }}. Keep it under 50 words.
    """
    pass

# Example 4: Using MetaHolder and return_origin_response
metadata_capture = MetaHolder()
@llm_client.prompt(meta_holder=metadata_capture, return_origin_response=True)
def summarize_text(text: str) -> str:
    """
    Summarize the following text in one sentence:
    {{ text }}
    """
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Call Example 1
    greeting = generate_greeting(name="Alice")
    print(f"Generated Greeting: {greeting}\n") # Output will be simulated

    # Call Example 2
    raw_text = "The item is called 'Super Widget' and its value is 123."
    extracted = extract_info(text=raw_text)
    if extracted:
        print(f"Extracted Info: Name={extracted.name}, Value={extracted.value}\n")
    else:
        print("Extraction failed or returned None.\n")


    # Call Example 3
    print("Streaming Story:")
    story_generator = stream_story(topic="a brave knight")
    for chunk in story_generator:
        print(chunk, end="", flush=True)
    print("\n--- End of Stream ---\n")

    # Call Example 4
    original_text = "The quick brown fox jumps over the lazy dog. This is a classic sentence used for testing typefaces."
    raw_responses = summarize_text(text=original_text)
    print(f"Raw LLM Response Object: {raw_responses[0].__dict__}") # Show the raw response object
    print(f"Captured Metadata: {metadata_capture.meta}\n") # Show the captured metadata
    print(f"Actual Summary Output: {raw_responses[0].output}\n")


```

## 依赖说明
- `pydantic`: 用于定义和解析结构化输出模型。
- `jinja2`: (可选，如果使用 `format_prompt_jinja2` 辅助函数) 用于渲染文档字符串模板。
- `loguru` or `logging`: 用于日志记录。
- `typing`, `inspect`, `functools`, `collections.abc`: Python 标准库。
- **核心依赖**: 一个实际的 LLM 客户端库 (如 `openai`)，你需要用它来实现 `chat_oai` 和 `stream_chat_oai` 方法。示例中使用的是占位符。
- **环境要求**: Python 3.7+ (为了类型注解和 Pydantic)。
- **初始化流程**:
    1. 实现或获取一个 LLM 客户端类，该类包含实际调用 LLM API 的方法（如 `chat_oai` 和 `stream_chat_oai`）。
    2. 将 `prompt` 装饰器方法添加到该客户端类中。
    3. (可选) 实现或导入 `format_prompt_jinja2` 辅助函数。
    4. 实例化客户端。
    5. 使用 `@client_instance.prompt(...)` 装饰器来定义你的提示函数。

## 学习来源
该模式提取自 `/Users/allwefantasy/projects/byzer-llm/src/byzerllm/utils/client/simple_byzerllm_client.py` 文件中的 `SimpleByzerLLM` 类的 `prompt` 方法及其相关逻辑。它封装了使用函数定义、文档字符串模板化、类型注解驱动的 API 调用和响应处理的流程。