import logging
from typing import Any, Callable, Type, Union

import openai
from openai import (
    Completion,
    ChatCompletion,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    APIError,
)

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

CompletionClientType = Union[Type[Completion], Type[ChatCompletion]]


def _create_retry_decorator(max_retries: int) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
                retry_if_exception_type(APITimeoutError)
                | retry_if_exception_type(APIError)
                | retry_if_exception_type(APIConnectionError)
                | retry_if_exception_type(RateLimitError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(is_chat_model: bool, max_retries: int, **kwargs: Any) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        client = get_completion_endpoint(is_chat_model)
        return client.create(**kwargs)

    return _completion_with_retry(**kwargs)


async def async_completion_with_retry(is_chat_model: bool, max_retries: int, **kwargs: Any) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(max_retries=max_retries)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        # Use OpenAI's async api https://github.com/openai/openai-python#async-api
        client = get_completion_endpoint(is_chat_model)
        return await client.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)


def get_completion_endpoint(is_chat_model: bool) -> CompletionClientType:
    if is_chat_model:
        return openai.ChatCompletion
    else:
        return openai.Completion

