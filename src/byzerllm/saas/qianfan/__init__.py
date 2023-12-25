import traceback
from typing import List, Dict

import qianfan


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.api_key: str = infer_params["saas.api_key"]
        self.secret_key: str = infer_params["saas.secret_key"]
        self.model: str = infer_params.get("saas.model", "ERNIE-Bot-turbo")
        self.endpoint: str = infer_params.get("saas.endpoint", None)
        self.request_timeout: float = infer_params.get("saas.request_timeout", 60)
        self.retry_count: int = infer_params.get("saas.retry_count", 5)
        self.backoff_factor: float = infer_params.get("saas.backoff_factor", 4)
        self.penalty_score: float = infer_params.get("penalty_score", 1.0)

        qianfan.AK(self.api_key)
        qianfan.SK(self.secret_key)

     # saas/proprietary
    def get_meta(self):
        return [{
            "model_deploy_type": "saas",
            "backend":"saas"
        }]    

    def stream_chat(
            self,
            tokenizer,
            ins: str,
            his: List[dict] = [],
            max_length: int = 4096,
            top_p: float = 0.7,
            temperature: float = 0.9,
            **kwargs
    ):
        timeout_s = kwargs.get("timeout_s", self.request_timeout)

        messages = qianfan.Messages()
        for item in his:
            role, content = item['role'], item['content']
            # messages must have an odd number of members
            # look for details: https://cloud.baidu.com/doc/WENXINWORKSHOP/s/clntwmv7t
            if role == 'system':
                messages.append(content, qianfan.Role.User)
                messages.append("OK", qianfan.Role.Assistant)
                continue
            messages.append(content, role)

        if ins:
            messages.append(ins, qianfan.Role.User)

        content = None
        try:
            client = qianfan.ChatCompletion()
            resp = client.do(
                model=self.model,
                endpoint=self.endpoint,
                retry_count=int(self.retry_count),
                request_timeout=float(timeout_s),
                backoff_factor=float(self.backoff_factor),
                penalty_score=float(self.penalty_score),
                messages=messages,
                top_p=top_p,
                temperature=temperature,
            )
            content = resp.body['result']
            print(f"【Qianfan({self.model}) --> Byzer】: {resp.body}")
        except Exception as e:
            traceback.print_exc()
            content = f"request qianfan api failed: {e}" if content is None or content == "" else content
        return [(content, "")]
