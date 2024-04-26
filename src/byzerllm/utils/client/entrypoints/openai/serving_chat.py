# Adapted from
# vLLM project
import asyncio
import time
from fastapi import Request
from typing import AsyncGenerator, Union, Optional

from byzerllm.log import init_logger
from byzerllm.utils.types import SingleOutputMeta
from byzerllm.utils.client.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ErrorResponse,
    UsageInfo
)

from byzerllm.utils import random_uuid
from byzerllm.utils.client import ByzerLLM, LLMResponse
from byzerllm.utils.client.entrypoints.openai.serving_engine import OpenAIServing

logger = init_logger(__name__)


class OpenAIServingChat(OpenAIServing):

    def __init__(
            self,
            llm_client: ByzerLLM,
            response_role: str,
            server_model_name: Optional[str] = None,
            prompt_template: Optional[str] = None
    ):
        super().__init__(
            llm_client=llm_client,
            server_model_name=server_model_name,
            prompt_template=prompt_template
        )
        self.response_role = response_role

    async def create_chat_completion(
            self, body: ChatCompletionRequest, request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None], ChatCompletionResponse]:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        ChatCompletion API.

        NOTE: Currently we do not support the following feature:
            - function_call (Users should implement this by themselves)
        """
        if body.prompt_template:
            self.llm_client.setup_template(body.model, self._detect_prompt_template(body.prompt_template))

        request_id = f"cmpl-{random_uuid()}"

        error_check_ret = await self._check_model(body)
        if error_check_ret is not None:
            return error_check_ret

        # Streaming response
        if body.stream:
            return self.chat_completion_stream_generator(body, request_id)
        else:
            try:
                return await self.chat_completion_full_generator(body, request, request_id)
            except ValueError as e:
                return self.create_error_response(str(e))

    def get_chat_request_role(self, body: ChatCompletionRequest) -> str:
        if body.add_generation_prompt:
            return self.response_role
        else:
            return body.messages[-1]["role"]

    async def chat_completion_stream_generator(
            self,
            body: ChatCompletionRequest,
            request_id: str
    ) -> Union[ErrorResponse, AsyncGenerator[str, None]]:
        model_name = body.model
        created_time = int(time.time())
        chunk_object_type = "chat.completion.chunk"

        result_generator = self.llm_client.async_stream_chat_oai(
            model=model_name,
            conversations=body.messages,
            delta_mode=True,
            llm_config={
                "gen.request_id": request_id,
                **body.to_llm_config()
            }
        )

        role = self.get_chat_request_role(body)

        for i in range(body.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i, delta=DeltaMessage(role=role), finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                choices=[choice_data],
                model=model_name
            )
            data = chunk.model_dump_json(exclude_unset=True)
            yield f"data: {data}\n\n"

        # Send response to echo the input portion of the last message
        if body.echo:
            last_msg_content = ""
            if (body.messages and isinstance(body.messages, list)
                    and body.messages[-1].get("content")
                    and body.messages[-1].get("role") == role):
                last_msg_content = body.messages[-1]["content"]
            if last_msg_content:
                for i in range(body.n):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=last_msg_content),
                        finish_reason=None
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name
                    )
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

        # Send response for each token for each request.n (index)
        finish_reason_sent = [False] * body.n
        async for (s, meta) in result_generator:
            meta: SingleOutputMeta
            for _ in [(s, meta)]:
                i = 0
                prompt_tokens = meta.input_tokens_count
                final_usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=meta.generated_tokens_count,
                    total_tokens=prompt_tokens + meta.generated_tokens_count,
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=i, delta=DeltaMessage(content=s), finish_reason=None
                )
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[choice_data],
                    model=model_name
                )
                if final_usage is not None:
                    chunk.usage = final_usage
                data = chunk.model_dump_json(
                    exclude_unset=True,
                    exclude_none=True,
                )
                yield f"data: {data}\n\n"
                finish_reason_sent[i] = True
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
            self,
            body: ChatCompletionRequest,
            request: Request,
            request_id: str
    ) -> Union[ErrorResponse, ChatCompletionResponse]:

        async def wrapper_chat_generator():
            r = self.llm_client.chat_oai(
                model=model_name,
                conversations=body.messages,
                llm_config={
                    "gen.request_id": request_id,
                    **body.to_llm_config()
                }
            )
            for _ in r:
                yield _

        result_generator = await asyncio.to_thread(wrapper_chat_generator)

        model_name = body.model
        created_time = int(time.time())
        final_res = None

        async for res in result_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.llm_client.abort(request_id, model=model_name)
                return self.create_error_response("Client disconnected")
            final_res = res
        assert final_res is not None

        choices = []
        role = self.get_chat_request_role(body)
        for res in [final_res]:
            res: LLMResponse
            choice_data = ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role=role, content=res.output),
                finish_reason=None,
            )
            choices.append(choice_data)

        if body.echo:
            last_msg_content = ""
            if (body.messages and isinstance(body.messages, list)
                    and body.messages[-1].get("content")
                    and body.messages[-1].get("role") == role):
                last_msg_content = body.messages[-1]["content"]

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = res.metadata.get("input_tokens_count", 0)
        num_generated_tokens = res.metadata.get("generated_tokens_count", 0)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response
