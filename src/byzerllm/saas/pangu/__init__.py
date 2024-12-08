import logging
import requests
import json
import traceback
from typing import List, Dict

import tenacity
from byzerllm.utils import generate_instruction_from_history
from tenacity import (
    before_sleep_log,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class PanguError(Exception):
    def __init__(
            self,
            request_id=None,
            error_msg=None,
            error_code=None,
            http_status_msg=None,
            http_status=None,
            http_body=None,
            headers=None,
    ):
        if http_body and hasattr(http_body, "decode"):
            try:
                http_body = http_body.decode("utf-8")
            except Exception as e:
                http_body = (
                    f"<Could not decode body as utf-8: {e}"
                )

        self._status_msg = http_status_msg
        self.http_body = http_body
        self.http_status = http_status
        self.headers = headers or {}
        self.request_id = request_id
        self.error_msg = error_msg
        self.error_code = error_code

    def __str__(self):
        msg = self.error_msg or "<empty message>"
        if self.request_id is not None:
            return "Request {0}: {1}".format(self.request_id, msg)
        elif self.error_code is not None:
            return "API return error, code: {0}, msg: {1}".format(self.http_status, msg)
        else:
            return msg


def _pangu_api_retry_if_need(exception):
    """
        CBS.0001: 请求体格式错误
        CBS.0011: auth failed
        CBS.0013: status prohibited
    """
    if isinstance(exception, PanguError):
        error_code = exception.error_code
        return (error_code == "CBS.3267"
                or error_code == "APIG.0201"
                or error_code == "APIG.0308")
    return False


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.is_chat_model = infer_params.get(
            "saas.is_chat_model", "true") == "true"
        self.api_token = infer_params["saas.api_token"]
        self.api_url = infer_params["saas.api_url"]

    def stream_chat(
            self,
            tokenizer,
            ins: str,
            his: List[Dict],
            max_length: int = 4096,
            top_p: float = 0.7,
            temperature: float = 0.9,
            **kwargs
    ):
        payload = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_length,
        }

        if not self.is_chat_model:
            role_mapping = {
                "user": "User",
                "assistant": "Assistant",
            }
            ins = generate_instruction_from_history(
                ins, his, role_mapping=role_mapping)
            payload['prompt'] = ins
        else:
            if ins != "":
                his.append({"role": "user", "content": ins})
            payload['messages'] = his

        print(f"【Byzer --> Pangu】: {payload}")

        content = None
        try:
            content = self.request_with_retry(payload)
        except Exception as e:
            traceback.print_exc()
            if content == "" or content is None:
                content = f"request pangu api failed: {e}"
        return [(content, "")]

    @tenacity.retry(
        reraise=True,
        retry=tenacity.retry_if_exception(_pangu_api_retry_if_need),
        stop=tenacity.stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def request_with_retry(self, payload):
        """Use tenacity to retry the completion call."""

        headers = {
            "X-Auth-Token": f"{self.api_token}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.api_url, data=json.dumps(
            payload), headers=headers, verify=False)
        if response.status_code == 200:
            res_data = json.loads(response.text)
            print(f"【Pangu --> Byzer】: {res_data}")
            if 'error_code' in res_data:
                raise PanguError(
                    request_id=res_data.get('id', None),
                    error_code=res_data.get('error_code', None),
                    error_msg=res_data.get('error_msg', None)
                )
            return res_data['choices'][0]['message']['content']
        else:
            print(
                f"request failed with status code `{response.status_code}`, "
                f"headers: `{response.headers}`, "
                f"body: `{response.content!r}`"
            )
            raise PanguError(
                headers=headers, http_status=response.status_code, http_body=response.text)
