import time
from typing import List, Dict, Any

from openai import AzureOpenAI

from byzerllm.log import init_logger
from byzerllm.utils import random_uuid

logger = init_logger(__name__)


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.model = infer_params["saas.model"]
        # gets the API Key from environment variable AZURE_OPENAI_API_KEY
        self.client = AzureOpenAI(
            # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
            api_version=infer_params["saas.api_version"],
            # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
            azure_endpoint=infer_params["saas.azure_endpoint"],
            azure_deployment=infer_params["saas.azure_deployment"],
            api_key=infer_params["saas.api_key"],
            max_retries=infer_params.get("saas.max_retries", 10)
        )

    # saas/proprietary
    def get_meta(self):
        return [{
            "model_deploy_type": "saas",
            "backend": "saas"
        }]

    def stream_chat(
            self,
            tokenizer,
            ins: str,
            his: List[Dict[str, Any]] = [],
            max_length: int = 4096,
            top_p: float = 0.7,
            temperature: float = 0.9,
            **kwargs
    ):
        request_id = random_uuid() if "request_id" not in kwargs else kwargs["request_id"]
        messages = his
        if ins:
            messages += [{"role": "user", "content": ins}]

        start_time = time.monotonic()

        answer = None
        try:
            logger.info(f"Receiving request {request_id}: model: {self.model} messages: {messages}")

            completion = self.client.chat.completions.create(
                model=self.model,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_length,
                messages=messages,
            )
            time_taken = time.monotonic() - start_time
            answer = completion.choices[0].message.content.strip()
            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens

            logger.info(
                f"Completed request {request_id}: "
                f"model: {self.model} "
                f"cost: {time_taken} "
                f"result: {completion.model_dump_json()}"
            )

            return [(
                answer,
                {
                    "metadata": {
                        "request_id": "",
                        "input_tokens_count": input_tokens,
                        "generated_tokens_count": output_tokens,
                        "time_cost": time_taken,
                        "first_token_time": -1.0,
                        "speed": float(output_tokens) / time_taken * 1000 if time_taken > 0 else 0,
                    }
                }
            )]
        except Exception as e:
            logger.error(f"request azure openai failed: {e}")
            answer = f"Exception occurred during the request, please try again: {e}" if not answer else answer
            return [(answer, "")]
