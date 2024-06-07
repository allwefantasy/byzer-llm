# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/serve/openai_api_server.py
# Adapted from
# vLLM project

import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional, Union

from byzerllm.log import init_logger
from byzerllm.utils.client import ByzerLLM, Templates
from byzerllm.utils.client.entrypoints.openai.protocol import (
    CompletionRequest,
    ChatCompletionRequest,
    ErrorResponse,
    LogProbs,
    ModelCard,
    ModelList,
    ModelPermission,
    Logprob
)

logger = init_logger(__name__)


@dataclass
class LoRA:
    name: str
    local_path: str


class OpenAIServing:

    def __init__(
            self,
            llm_client: ByzerLLM,
            server_model_name: Optional[str] = None,
            prompt_template: Optional[str] = None
    ):
        self.llm_client = llm_client
        self.max_model_len = 0
        self.tokenizer = None
        self.server_model_name = server_model_name
        if server_model_name and prompt_template:
            self.llm_client.setup_template(
                server_model_name, self._detect_prompt_template(prompt_template)
            )

    async def show_available_models(self) -> ModelList:
        """Show available models. Right now we only have one model."""
        model_cards = [
            ModelCard(
                id="",
                root="",
                permission=[ModelPermission()]
            )
        ]
        return ModelList(data=model_cards)

    def _create_logprobs(
            self,
            token_ids: List[int],
            top_logprobs: Optional[List[Optional[Dict[int, Logprob]]]] = None,
            num_output_top_logprobs: Optional[int] = None,
            initial_text_offset: int = 0,
    ) -> LogProbs:
        """Create OpenAI-style logprobs."""
        logprobs = LogProbs()
        last_token_len = 0
        if num_output_top_logprobs:
            logprobs.top_logprobs = []
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is not None:
                token_logprob = step_top_logprobs[token_id].logprob
            else:
                token_logprob = None
            token = step_top_logprobs[token_id].decoded_token
            logprobs.tokens.append(token)
            logprobs.token_logprobs.append(token_logprob)
            if len(logprobs.text_offset) == 0:
                logprobs.text_offset.append(initial_text_offset)
            else:
                logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
            last_token_len = len(token)

            if num_output_top_logprobs:
                logprobs.top_logprobs.append(
                    {
                        p.decoded_token: p.logprob for i, p in step_top_logprobs.items()
                    } if step_top_logprobs else None
                )
        return logprobs

    def create_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        return ErrorResponse(
            message=message,
            type=err_type,
            code=status_code.value
        )

    def create_streaming_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> str:
        json_str = json.dumps({
            "error": self.create_error_response(
                message=message,
                err_type=err_type,
                status_code=status_code
            ).model_dump()
        })
        return json_str

    async def _check_model(self, body) -> Optional[ErrorResponse]:
        if  self.server_model_name or self.llm_client.is_model_exist(body.model):
            return
        return self.create_error_response(
            message=f"The model `{body.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND
        )

    def _validate_prompt_and_tokenize(
            self,
            request: Union[ChatCompletionRequest, CompletionRequest],
            prompt: Optional[str] = None,
            prompt_ids: Optional[List[int]] = None) -> List[int]:
        if not (prompt or prompt_ids):
            raise ValueError("Either prompt or prompt_ids should be provided.")
        if (prompt and prompt_ids):
            raise ValueError(
                "Only one of prompt or prompt_ids should be provided.")

        input_ids = prompt_ids if prompt_ids is not None else self.tokenizer(
            prompt).input_ids
        token_num = len(input_ids)

        if request.max_tokens is None:
            request.max_tokens = self.max_model_len - token_num

        if token_num + request.max_tokens > self.max_model_len:
            raise ValueError(
                f"This model's maximum context length is "
                f"{self.max_model_len} tokens. However, you requested "
                f"{request.max_tokens + token_num} tokens "
                f"({token_num} in the messages, "
                f"{request.max_tokens} in the completion). "
                f"Please reduce the length of the messages or completion.", )
        else:
            return input_ids

    def _detect_prompt_template(self, tpl: str):
        if tpl == "qwen":
            return Templates.qwen()
        elif tpl == "yi":
            return Templates.yi()
        elif tpl == "llama":
            return Templates.llama()
        elif tpl == "empty":
            return Templates.empty()
        elif tpl == "auto":
            return "auto"
        else:
            logger.warning(
                f"Prompt template {tpl} was not found and auto template was used."
            )
            return "auto"
