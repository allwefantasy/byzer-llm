import time
import requests
import base64

from zhipuai import ZhipuAI
from typing import Dict, List, Any

from byzerllm.utils.langutil import asyncfy_with_semaphore


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.api_key = infer_params["saas.api_key"]
        self.model = infer_params.get("saas.model", "cogView-3-plus")
        self.client = ZhipuAI(api_key=self.api_key)

        self.meta = {
            "model_deploy_type": "saas",
            "backend": "saas",
            "support_stream": False,
            "embedding_mode": False
        }

        self.valid_sizes = infer_params.get("saas.valid_sizes", [
            "1024x1024",
            "768x1344",
            "864x1152",
            "1344x768",
            "1152x864",
            "1440x720",
            "720x1440"
        ])
    def get_meta(self):
        return [self.meta]

    async def async_get_meta(self):
        return await asyncfy_with_semaphore(self.get_meta)()

    def stream_chat(
            self,
            tokenizer,
            ins: str,
            his: List[Dict[str, Any]],
            size: str = "1024x1024",
            quality: str = "standard",
            n: int = 1,
            **kwargs
    ):
        try:
            print(f"【Byzer --> ZhipuAI({self.model})】, size:[{size}], quality:[{quality}], n:[{n}]")
            if self.valid_sizes and size not in self.valid_sizes:
                raise ValueError(f"Invalid size parameter [{size}]. Must be one of: {', '.join(self.valid_sizes)}")

            start_time = time.monotonic()
            response = self.client.images.generations(
                model=self.model,
                prompt=ins,
                size=size,
                quality=quality,
                n=n,
            )
            time_cost = time.monotonic() - start_time
            image_url = response.data[0].url

            if image_url is None:
                raise Exception("Failed to generate image")

            response = requests.get(image_url)
            image_data = response.content
            base64_image = base64.b64encode(image_data).decode('utf-8')

            print(f"【Byzer --> ZhipuAI({self.model})】: Generated image url: {image_url}")
            return [
                (
                    base64_image,
                    {
                        "metadata": {
                            "request_id": "",
                            "input_tokens_count": 0,
                            "generated_tokens_count": 0,
                            "time_cost": time_cost,
                            "first_token_time": 0,
                            "speed": 0,
                        }
                    },
                )
            ]
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Request ZhipuAI API failed: {str(e)}"
            print(f"【Error】: {error_msg}")
            raise Exception(error_msg)
