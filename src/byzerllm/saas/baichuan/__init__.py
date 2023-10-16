import requests
import json
import time
import hashlib
import traceback
from retrying import retry
from typing import List, Tuple, Dict

BaiChuanErrorCodes = {
    "0": "success",
    "1": "system error",
    "10000": "Invalid parameters, please check",
    "10100": "Missing apikey",
    "10101": "Invalid apikey",
    "10102": "apikey has expired",
    "10103": "Invalid Timestamp parameter in request header",
    "10104": "Expire Timestamp parameter in request header",
    "10105": "Invalid Signature parameter in request header",
    "10106": "Invalid encryption algorithm in request header, not supported by server",
    "10200": "Account not found",
    "10201": "Account is locked, please contact the support staff",
    "10202": "Account is temporarily locked, please try again later",
    "10203": "Request too frequent, please try again later",
    "10300": "Insufficient account balance, please recharge",
    "10301": "Account is not verified, please complete the verification first",
    "10400": "Topic violates security policy for prompts",
    "10401": "Topic violates security policy for answer",
    "10500": "Internal error",
}


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.api_key = infer_params["saas.api_key"]
        self.secret_key = infer_params["saas.secret_key"]
        self.api_url = infer_params.get("saas.baichuan_api_url", "https://api.baichuan-ai.com/v1/chat")
        self.model = infer_params.get("saas.model", "Baichuan2-53B")

    def stream_chat(self, tokenizer, ins: str, his: List[Tuple[str, str]] = [],
                    max_length: int = 4096,
                    top_p: float = 0.7,
                    temperature: float = 0.9, **kwargs):

        his_message = []

        for item in his:
            his_message.append({"role": "user", "content": item[0]})
            his_message.append({"role": "assistant", "content": item[1]})
        messages = his_message + [{"role": "user", "content": ins}]

        data = {
            "model": self.model,
            "messages": messages
        }

        content = None

        try:
            content = self.request_with_retry(data)
        except Exception as e:
            traceback.print_exc()
            if content == "" or content is None:
                content = f"exception occurred during the request, please check the error code: {e}"
        return [(content, "")]

    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=10)
    def request_with_retry(self, data):
        json_data = json.dumps(data)
        time_stamp = int(time.time())
        signature = self.calculate_md5(self.secret_key + json_data + str(time_stamp))
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "X-BC-Timestamp": str(time_stamp),
            "X-BC-Signature": signature,
            "X-BC-Sign-Algo": "MD5",
        }
        response = requests.post(self.api_url, data=json_data, headers=headers)
        if response.status_code == 200:
            print("request baichuan api success")
            print("response text: " + response.text)
            res_data = json.loads(response.text)
            if res_data["code"] != 0:
                if str(res_data["code"]) in BaiChuanErrorCodes.keys():
                    msg = BaiChuanErrorCodes.get(str(res_data["code"]))
                    print("request baichuan api failed, err msg:" + msg)
                    # check if api call over limit
                    if str(res_data["code"]) == "10203":
                        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        print("{0} request baichuan api too frequent, retrying... ".format(now))
                        raise Exception("request baichuan api too frequent." + str(res_data["code"]))
                    else:
                        return msg
                # unexpected error code, retry anyway
                raise Exception("request baichuan api failed, api response code:" + str(res_data["code"]))
            content = res_data["data"]["messages"][0]["content"].replace(' .', '.').strip()
            return content
        else:
            print("request baichuan api failed, http response code:" + str(response.status_code))
            print("response text:" + response.text)
            raise Exception("request baichuan api failed")

    def calculate_md5(self, input_string):
        md5 = hashlib.md5()
        md5.update(input_string.encode('utf-8'))
        encrypted = md5.hexdigest()
        return encrypted

