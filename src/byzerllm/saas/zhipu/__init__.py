import json
import zhipuai
import time
import traceback
from typing import List, Tuple, Dict,Any
from retrying import retry


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.api_key = infer_params["saas.api_key"]
        # chatglm_lite, chatglm_std, chatglm_pro
        self.model = infer_params.get("saas.model", "chatglm_lite")
        self.temperature = infer_params.get("saas.model.temperature", "0.7")
        self.topP = infer_params.get("saas.model.topp", "0.9")
        zhipuai.api_key = self.api_key

    @retry(wait_exponential_multiplier=5000, wait_exponential_max=30000, stop_max_attempt_number=4)
    def request_with_retry(self, messages: []):
        response = zhipuai.model_api.invoke(
            model=self.model,
            prompt=messages,
            top_p=float(self.topP),
            temperature=float(self.temperature),
	    return_type="text"	
        )
        if response["code"] == 200:
            content = response["data"]["choices"][0]["content"].replace(' .', '.').strip('"').strip()
            return content
        else:
            # for over limit call, we need retry
            if response["code"] in [1302, 1303, 1305]:
                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print("{0} request zhipu api too frequent, retrying... ".format(now))
                raise Exception("call zhipu api too frequent.")
            msg = response["msg"]
            print("request zhipu api failed, response code:" + str(response["code"]))
            print("response:" + json.dumps(response))
            return msg

    # saas/proprietary
    def get_meta(self):
        return [{
            "model_deploy_type": "saas",
            "backend":"saas"
        }]    

    def stream_chat(self, tokenizer, ins: str, his: List[Dict[str, Any]] = [],
                    max_length: int = 4096,
                    top_p: float = 0.7,
                    temperature: float = 0.9, **kwargs):
        
        messages = his + [{"role": "user", "content": ins}]

        zhipuai.api_key = self.api_key

        content = None

        try:
            content = self.request_with_retry(messages)
        except Exception as e:
            traceback.print_exc()
            if content == "" or content is None:
                content = "exception occurred during the request, please check the error code"
        return [(content, "")]

