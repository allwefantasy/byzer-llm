# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/serve/openai_api_server.py
# Adapted from
# vLLM project

import argparse
import asyncio
import codecs
import json
import time
import ray
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

from aioprometheus import MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response

from byzerllm.utils.client import ByzerLLM,LLMResponse
from byzerllm.utils import SingleOutputMeta

from byzerllm.utils.client.entrypoints.openai.protocol import (
    CompletionRequest, CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    LogProbs, ModelCard, ModelList, ModelPermission, UsageInfo)

import logging
import uuid

def random_uuid() -> str:
    return str(uuid.uuid4().hex)



TIMEOUT_KEEP_ALIVE = 5  # seconds
logger = logging.getLogger(__name__)

ray.init(address = "auto", namespace="default",ignore_reinit_error=True)

app = fastapi.FastAPI()
response_role = "assistant"


def parse_args():
    parser = argparse.ArgumentParser(
        description="ByzerLLm OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")            
    parser.add_argument("--ssl-keyfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL key file")
    parser.add_argument("--ssl-certfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL cert file")
    
    return parser.parse_args()


app.add_middleware(MetricsMiddleware)  # Trace HTTP server metrics
app.add_route("/metrics", metrics)  # Exposes HTTP metrics


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message,
                                      type="invalid_request_error").dict(),
                        status_code=status_code.value)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))



@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id="",
                  root="",
                  permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """    

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")       

    model_name = request.model
    
    llm = ByzerLLM()
    llm.setup_template(model=model_name,template="auto")
    llm.setup_extra_generation_params(model_name,extra_generation_params={
      "temperature":request.temperature or 0.01,
      "top_p":request.top_p or 0.99
    })

    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())
    chunk_object_type = "chat.completion.chunk"            
    
    
    def get_role() -> str:
        if request.add_generation_prompt:
            return response_role
        else:
            return request.messages[-1]["role"]
        

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # Send first response for each request.n (index) with the role          
        result_generator = llm.async_stream_chat_oai(model=model_name,
                                           conversations=request.messages,
                                           llm_config={"gen.request_id":request_id}) 
        role = get_role()     
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i, delta=DeltaMessage(role=role), finish_reason=None)
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                 object=chunk_object_type,
                                                 created=created_time,
                                                 choices=[choice_data],
                                                 model=model_name)
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
            yield f"data: {data}\n\n"

        # Send response to echo the input portion of the last message
        if request.echo:
            last_msg_content = ""
            if request.messages and isinstance(
                    request.messages, list) and request.messages[-1].get(
                        "content") and request.messages[-1].get(
                            "role") == role:
                last_msg_content = request.messages[-1]["content"]
            if last_msg_content:
                for i in range(request.n):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=last_msg_content),
                        finish_reason=None)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)
                    data = chunk.json(exclude_unset=True, ensure_ascii=False)
                    yield f"data: {data}\n\n"

        # Send response for each token for each request.n (index)        
        finish_reason_sent = [False] * request.n
        async for (s,meta) in result_generator:  
            meta: SingleOutputMeta          
            for _ in [(s,meta)]:
                i = 0                              
                prompt_tokens = meta.input_tokens_count
                final_usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=meta.generated_tokens_count,
                    total_tokens=prompt_tokens + meta.generated_tokens_count,
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=i, delta=DeltaMessage(content=s), finish_reason=None)
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[choice_data],
                    model=model_name)
                if final_usage is not None:
                    chunk.usage = final_usage
                data = chunk.json(exclude_unset=True,
                                    exclude_none=True,
                                    ensure_ascii=False)
                yield f"data: {data}\n\n"
                finish_reason_sent[i] = True
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def completion_full_generator():
        
        async def chat():
            r = llm.chat_oai(model=model_name,
                            conversations=request.messages,
                            llm_config={"gen.request_id":request_id}) 
            for _ in r:
                yield _
        
        result_generator = await asyncio.to_thread(chat) 
        
        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await llm.abort(request_id,model=model_name)
                return create_error_response(HTTPStatus.BAD_REQUEST,
                                             "Client disconnected")
            final_res = res
        assert final_res is not None

        choices = []
        role = get_role()
        for r in [final_res]:
            r:LLMResponse
            choice_data = ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role=role, content=r.output),
                finish_reason=None,
            )
            choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if request.messages and isinstance(
                    request.messages, list) and request.messages[-1].get(
                        "content") and request.messages[-1].get(
                            "role") == role:
                last_msg_content = request.messages[-1]["content"]

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = r.metadata["input_tokens_count"]
        num_generated_tokens = r.metadata["generated_tokens_count"]

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

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream")
    else:
        return await completion_full_generator()


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.

    NOTE: Currently we do not support the following features:
        - suffix (the language models we currently support do not support
          suffix)
        - logit_bias (to be supported by vLLM engine)
    """
    

    # OpenAI API supports echoing the prompt when max_tokens is 0.
    echo_without_generation = request.echo and request.max_tokens == 0   

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    
    created_time = int(time.monotonic())  
    
    # Similar to the OpenAI API, when n != best_of, we do not stream the
    # results. In addition, we do not stream the results when use beam search.
    stream = request.stream

    def create_stream_response_json(
        index: int,
        text: str,
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
            model=model_name,
            choices=[choice_data],
        )
        if usage is not None:
            response.usage = usage
        response_json = response.json(exclude_unset=True, ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        previous_texts = [""] * request.n    
        result_generator = llm.async_stream_chat_oai(model=model_name,
                                           conversations=[{
                                                  "role": "user",
                                                  "content": request.prompt
                                           }],
                                           llm_config={"gen.request_id":request_id})

        async for res in result_generator:            
            (s,meta) = res
            meta:SingleOutputMeta
            for _ in [(s,meta)]:
                i = 0
                delta_text = s[len(previous_texts[i]):]                
                top_logprobs = None
                logprobs = None                            
                
                previous_texts[i] = s                
                finish_reason = None
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                    logprobs=logprobs,
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
                response_json = create_stream_response_json(
                    index=i,
                    text="",
                    logprobs=logprobs,
                    finish_reason=None,
                    usage=final_usage,
                )
                yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    # Streaming response
    if stream:
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream")

    # Non-streaming response
    async def chat():
            r = llm.chat_oai(model=model_name,
                            conversations=request.messages,
                            llm_config={"gen.request_id":request_id}) 
            for _ in r:
                yield _
        
    result_generator = await asyncio.to_thread(chat) 
    final_res = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await llm.abort(request_id,model=model_name)
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
        
    
    for r in [final_res]:  
        r:LLMResponse     
        choice_data = CompletionResponseChoice(
            index=0,
            text=r.input,
            logprobs=None,
            finish_reason=None,
        )
        choices.append(choice_data)

    num_prompt_tokens = r.metadata["input_tokens_count"]
    num_generated_tokens = r.metadata["generated_tokens_count"]
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

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response


if __name__ == "__main__":
    args = parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")
    
    # Register labels for metrics
    # add_global_metrics_labels(model_name=engine_args.model)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
