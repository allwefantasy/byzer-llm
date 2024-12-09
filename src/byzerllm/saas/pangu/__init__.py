import requests
import json
import traceback
import time
from typing import List, Dict, Any, Any

from byzerllm.log import init_logger
from byzerllm.utils import random_uuid

logger = init_logger(__name__)


DEFAULT_ROLE_MAPPING = {
    "user": "User: ",
    "assistant": "Assistant: ",
    "system": ""
}


def _convert_pangu_ai_messages(ins: str, his: List[Dict[str, Any]]):
    last_role = "user"
    final_message = ""

    if ins:
        his += [{"role": "user", "content": ins}]

    for msg in his:
        role, content = msg['role'], msg['content']
        role_prefix = DEFAULT_ROLE_MAPPING.get(role, f"{role}: ")
        final_message += f"{role_prefix}{content}\n"

    if last_role == "user":
        final_message += DEFAULT_ROLE_MAPPING.get("assistant")

    return [final_message]


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.model = infer_params.get("saas.model", "pangu-nlp-71b")
        self.api_url = infer_params.get("saas.api_url")

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
            his: List[Dict[str, Any]],
            max_length: int = 4096,
            top_p: float = 0.7,
            temperature: float = 0.9,
            **kwargs
    ):
        request_id = random_uuid(
        ) if "request_id" not in kwargs else kwargs["request_id"]

        messages = _convert_pangu_ai_messages(ins, his)

        payload = {
            "lang": kwargs.get("lang", "cn"),
            "data": messages,
            "decode_strategy": {
                "temperature": max(temperature, 0.001),
                "max_output_tokens": max_length,
                "top_p": top_p
            }
        }

        logger.info(
            f"Receiving request {request_id}: model: {self.model} payload: {payload}"
        )
        content = None
        try:
            return self._do_request(payload, request_id)
        except Exception as e:
            traceback.print_exc()
            content = f"request pangu api failed: {e}"
        return [(content, "")]

    def _do_request(self, payload, request_id):
        start_time = time.monotonic()
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(
            self.api_url, data=json.dumps(payload), headers=headers,verify=False
        )
        if response.status_code == 200:
            res_data = json.loads(response.text)

            time_taken = time.monotonic() - start_time
            answer = res_data["answers"][0]["content"]
            input_tokens = res_data["tokens"] - res_data["answers"][0]["tokens"]
            output_tokens = res_data["answers"][0]["tokens"]

            logger.info(
                f"Completed request {request_id}: "
                f"model: {self.model} "
                f"cost: {time_taken} "
                f"result: {res_data}"
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
        else:
            raise Exception(
                f"request failed with status code `{response.status_code}`, "
                f"headers: `{response.headers}`, "
                f"body: `{response.content!r}`"
            )