# Adapted from
# vLLM project

import asyncio
import time
from typing import AsyncGenerator, Optional

from fastapi import Request

from byzerllm.utils.types import SingleOutputMeta
from byzerllm.utils import random_uuid
from byzerllm.utils.client import ByzerLLM, LLMResponse
from byzerllm.utils.client.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    LogProbs,
    UsageInfo,
)
from byzerllm.utils.client.entrypoints.openai.serving_engine import OpenAIServing


class OpenAIServingCompletion(OpenAIServing):

    def __init__(
            self,
            llm_client: ByzerLLM,
            server_model_name: Optional[str] = None,
            prompt_template: Optional[str] = None
    ):
        super().__init__(
            llm_client=llm_client,
            server_model_name=server_model_name,
            prompt_template=prompt_template
        )

    async def create_completion(
            self,
            body: CompletionRequest,
            request: Request
    ):
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/completions/create
        for the API specification. This API mimics the OpenAI Completion API.

        NOTE: Currently we do not support the following feature:
            - suffix (the language models we currently support do not support
            suffix)
        """

        # OpenAI API supports echoing the prompt when max_tokens is 0.
        echo_without_generation = body.echo and body.max_tokens == 0

        if body.prompt_template:
            self.llm_client.setup_template(body.model, self._detect_prompt_template(body.prompt_template))

        model_name = self.server_model_name or body.model
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.monotonic())

        # Similar to the OpenAI API, when n != best_of, we do not stream the
        # results. In addition, we do not stream the results when use beam search.
        if body.stream:
            return self.completion_stream_generator(body, request_id, created_time)

        # Non-streaming response
        async def wrapper_chat_generator():
            r = self.llm_client.chat_oai(
                model=model_name,
                conversations=[
                    {
                        "role": "user",
                        "content": body.prompt
                    }
                ],
                llm_config={
                    "gen.request_id": request_id,
                    **body.to_llm_config()
                }
            )
            for _ in r:
                yield _

        result_generator = await asyncio.to_thread(wrapper_chat_generator)
        final_res = None
        async for res in result_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.llm_client.abort(request_id, model=model_name)
                return self.create_error_response("Client disconnected")
            final_res = res
        assert final_res is not None
        choices = []

        for r in [final_res]:
            r: LLMResponse
            choice_data = CompletionResponseChoice(
                index=0,
                text=r.output,
                logprobs=None,
                finish_reason=None,
            )
            choices.append(choice_data)

        num_prompt_tokens = r.metadata.get("input_tokens_count", 0)
        num_generated_tokens = r.metadata.get("generated_tokens_count", 0)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        if body.stream:
            # When user requests streaming but we don't stream, we still need to
            # return a streaming response with a single event.
            response_json = response.json(ensure_ascii=False)

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            return fake_stream_generator()

        return response

    def create_stream_response_json(
            self,
            body: CompletionRequest,
            index: int,
            text: str,
            request_id: str,
            created_time: int,
            logprobs: Optional[LogProbs] = None,
            finish_reason: Optional[str] = None,
            usage: Optional[UsageInfo] = None,
    ) -> str:
        choice_data = CompletionResponseStreamChoice(
            index=index,
            text=text,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )
        response = CompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=body.model,
            choices=[choice_data],
        )
        if usage is not None:
            response.usage = usage
        response_json = response.model_dump_json(exclude_unset=True)

        return response_json

    async def completion_stream_generator(
            self,
            body: CompletionRequest,
            request_id: str,
            created_time: int
    ) -> AsyncGenerator[str, None]:
        model_name = self.server_model_name or body.model
        previous_texts = [""] * body.n
        result_generator = self.llm_client.async_stream_chat_oai(
            model=model_name,
            conversations=[{
                "role": "user",
                "content": body.prompt
            }],
            delta_mode=True,
            llm_config={
                "gen.request_id": request_id,
                **body.to_llm_config()
            }
        )

        async for res in result_generator:
            (s, meta) = res
            meta: SingleOutputMeta
            for _ in [(s, meta)]:
                i = 0
                delta_text = s[len(previous_texts[i]):]
                top_logprobs = None
                logprobs = None

                previous_texts[i] = s
                finish_reason = None
                response_json = self.create_stream_response_json(
                    body=body,
                    index=i,
                    text=delta_text,
                    logprobs=logprobs,
                    request_id=request_id,
                    created_time=created_time,
                    finish_reason=finish_reason,
                )
                yield f"data: {response_json}\n\n"
                completion_tokens = meta.generated_tokens_count
                prompt_tokens = meta.input_tokens_count
                final_usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )
                response_json = self.create_stream_response_json(
                    body=body,
                    index=i,
                    text="",
                    request_id=request_id,
                    created_time=created_time,
                    logprobs=logprobs,
                    finish_reason=None,
                    usage=final_usage,
                )
                yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"
