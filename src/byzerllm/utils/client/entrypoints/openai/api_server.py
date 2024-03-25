import argparse
import asyncio
import json
import os
import time

import ray
import async_timeout
import uvicorn
from fastapi import Request
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response

from byzerllm.log import logger
from byzerllm.utils import random_uuid
from byzerllm.version import __version__ as version
from byzerllm.utils.client import ByzerLLM, LLMRequest
from byzerllm.utils.client.entrypoints.openai.serving_chat import OpenAIServingChat
from byzerllm.utils.client.entrypoints.openai.serving_completion import OpenAIServingCompletion
from byzerllm.utils.client.entrypoints.openai.protocol import (
    ModelList,
    ModelCard,
    ModelPermission,
    ChatCompletionRequest,
    ErrorResponse,
    CompletionRequest,
    Embeddings,
    EmbeddingsOutput,
    EmbeddingsData,
    EmbeddingsUsage,
)

llm_client: ByzerLLM = None
openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None

TIMEOUT_KEEP_ALIVE = 5  # seconds
# timeout in 10 minutes. Streaming can take longer than 3 min
TIMEOUT = float(os.environ.get("BYZERLLM_APISERVER_HTTP_TIMEOUT", 600))


def init() -> FastAPI:
    router_app = FastAPI()

    # router_app.add_exception_handler(OpenAIHTTPException, openai_exception_handler)
    # router_app.add_exception_handler(HTTPException, openai_exception_handler)
    # router_app.add_exception_handler(HTTPXHTTPStatusError, openai_exception_handler)
    router_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return router_app


router_app = init()


@router_app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@router_app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@router_app.get("/version")
async def show_version():
    return JSONResponse(content={"version": version})


@router_app.get("/v1/models", response_model=ModelList)
async def models() -> ModelList:
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(
            id="",
            root="",
            permission=[ModelPermission()]
        )
    ]
    return ModelList(data=model_cards)


@router_app.post("/v1/completions")
async def create_completion(
        body: CompletionRequest,
        request: Request
):
    generator = await openai_serving_completion.create_completion(body, request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(),
            status_code=generator.code
        )
    if request.stream:
        return StreamingResponse(
            content=generator,
            media_type="text/event-stream"
        )
    else:
        return JSONResponse(content=generator.model_dump())


@router_app.post("/v1/chat/completions")
async def create_chat_completion(
        body: ChatCompletionRequest,
        request: Request,
):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """
    # async with async_timeout.timeout(TIMEOUT):

    generator = await openai_serving_chat.create_chat_completion(body, request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(),
            status_code=generator.code
        )
    if request.stream:
        return StreamingResponse(
            content=generator,
            media_type="text/event-stream"
        )
    else:
        return JSONResponse(content=generator.model_dump())


@router_app.post("/v1/embeddings")
async def embed(
        body: Embeddings,
        request: Request,
):
    """Given a prompt, the model will return one embedding.

    Returns:
        A response object with an embedding.
    """
    embedding_id = f"embed-{random_uuid()}"

    async with async_timeout.timeout(TIMEOUT):
        if isinstance(body.input, str):
            input = [body.input]
        else:
            input = body.input
        embed_prompts = [LLMRequest(instruction=x) for x in input]
        results_list = await asyncio.gather(
            *[
                llm_client.emb(body.model, prompt)
                for prompt in embed_prompts
            ]
        )
        final_results = []
        tokens = 0
        for results in results_list:
            # if results.error:
            #     raise OpenAIHTTPException.from_model_response(results)
            final_results.append(results.dict())
            tokens += results.num_input_tokens

        return EmbeddingsOutput(
            data=[
                EmbeddingsData(
                    embedding=results["embedding_outputs"],
                    index=i,
                    object="embedding",
                )
                for i, results in enumerate(final_results)
            ],
            id=embedding_id,
            object="list",
            created=int(time.time()),
            model=body.model,
            usage=EmbeddingsUsage(
                prompt_tokens=tokens,
                total_tokens=tokens,
            ),
        )


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
    parser.add_argument("--response-role",
                        type=str,
                        default="assistant",
                        help="The role name to return if "
                             "`request.add_generation_prompt=true`.")
    parser.add_argument("--ssl-keyfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL key file")
    parser.add_argument("--ssl-certfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL cert file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"ByzerLLM API server version {version}")
    logger.info(f"args: {args}")

    ray.init(
        "auto", namespace="default", ignore_reinit_error=True
    )

    # Register labels for metrics
    # add_global_metrics_labels(model_name=engine_args.model)
    llm_client = ByzerLLM()

    openai_serving_chat = OpenAIServingChat(
        llm_client=llm_client,
        response_role=args.response_role
    )

    openai_serving_completion = OpenAIServingCompletion(
        llm_client=llm_client,
        response_role=args.response_role
    )

    uvicorn.run(
        router_app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile
    )
