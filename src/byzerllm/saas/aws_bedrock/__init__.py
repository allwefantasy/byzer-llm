import json
import time
from abc import ABC
from typing import List, Tuple, Dict, Optional

import boto3

from byzerllm.log import init_logger
from byzerllm.utils import random_uuid
from byzerllm.utils.langutil import asyncfy_with_semaphore

logger = init_logger(__name__)


class SupportModelProviders:
    Anthropic = "anthropic"
    Meta = "meta"


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.aws_access_key = infer_params["saas.aws_access_key"]
        self.aws_secret_key = infer_params["saas.aws_secret_key"]
        self.region_name = infer_params["saas.region_name"]
        self.model = infer_params["saas.model"]
        self.model_api_version = infer_params.get("saas.model_api_version", None)
        self.model_provider = self.model.split(".")[0]

        if self.model_provider == SupportModelProviders.Anthropic and self.model_api_version is None:
            self.model_api_version = "bedrock-2023-05-31"

        self.model_provider = AwsBedrockModelFactory.get_bedrock_model(
            model_provider=self.model_provider,
            model_api_version=self.model_api_version,
            model_id=self.model,
            aws_access_key=self.aws_access_key,
            aws_secret_key=self.aws_secret_key,
            region_name=self.region_name
        )

    # saas/proprietary
    def get_meta(self):
        return [{
            "model_deploy_type": "saas",
            "backend": "saas",
        }]
    
    async def async_get_meta(self):
        return await asyncfy_with_semaphore(self.get_meta)()
    
    async def async_stream_chat(
            self,
            tokenizer,
            ins: str,
            his: List[Tuple[str, str]] = [],
            max_length: int = 4096,
            top_p: float = 0.7,
            temperature: float = 0.9,
            **kwargs
    ):
        return await asyncfy_with_semaphore(self.stream_chat)(tokenizer, ins, his, max_length, top_p, temperature, **kwargs)

    def stream_chat(
            self,
            tokenizer,
            ins: str,
            his: List[Tuple[str, str]] = [],
            max_length: int = 4096,
            top_p: float = 0.7,
            temperature: float = 0.9,
            **kwargs
    ):
        try:
            return self.model_provider.generate(
                ins=ins,
                his=his,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"request aws bedrock failed: {e}")
        return None


class AWSBedrockModel(ABC):
    bedrock_client = None
    model_id = None
    model_api_version = None

    def __init__(
            self,
            model_id: str,
            region_name: str = None,
            aws_access_key: str = None,
            aws_secret_key: str = None,
            model_api_version: Optional[str] = None,

    ):
        self.model_id = model_id
        self.model_api_version = model_api_version
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name,
        )

    def generate(
            self,
            ins: str,
            his: List[Tuple[str, str]] = [],
            max_length: int = 4096,
            top_p: float = 0.7,
            temperature: float = 0.9,
            **kwargs):
        pass


class AnthropicModel(AWSBedrockModel):

    def generate(
            self,
            ins: str,
            his: List[Tuple[str, str]] = [],
            max_length: int = 4096,
            top_p: float = 0.7,
            temperature: float = 0.9,
            **kwargs
    ):
        body = self._generate_anthropic_request_body(ins, his)

        logger.info(f"【Byzer --> AWS-Bedrock({self.model_id})】:\n{body}")

        answer = None
        input_tokens = 0
        output_tokens = 0
        time_taken = 0
        start_time = time.monotonic()

        try:
            api_res = self.bedrock.invoke_model(
                body=json.dumps(body),
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            time_taken = time.monotonic() - start_time
            response_body = json.loads(api_res.get('body').read())
            logger.info(
                f"【AWS-Bedrock({self.model_id}) --> Byzer】:\n{response_body}"
            )
            answer = response_body['content'][0]['text']
            input_tokens = response_body['usage']['input_tokens']
            output_tokens = response_body['usage']['output_tokens']
        except Exception as e:
            logger.error(f"request aws bedrock failed: {e}")
            answer = f"Exception occurred during the request, please try again: {e}" if not answer else answer

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

    def _generate_anthropic_request_body(
            self,
            ins: str,
            his: List[Tuple[str, str]] = [],
            max_length: int = 4096,
            temperature: float = 0.9,
    ):
        messages = []
        for item in his:
            role, content = item["role"], item["content"]
            if role == "system":
                messages.append({"role": "user", "content": [{"type": "text", "text": content}]})
                messages.append({"role": "assistant", "content": [{"type": "text", "text": "OK"}]})
                continue
            messages.append({"role": role, "content": [{"type": "text", "text": content}]})

        if ins and len(messages) == 0:
            messages.append({"role": "user", "content": [{"type": "text", "text": ins}]})

        if ins and len(messages) > 0:
            last_message_role = messages[-1]["role"]
            if last_message_role == "user":
                messages.append({"role": "assistant", "content": [{"type": "text", "text": "OK"}]})
            messages.append({"role": "user", "content": [{"type": "text", "text": ins}]})

        anthropic_request_body = {
            "anthropic_version": self.model_api_version,
            "temperature": temperature,
            "max_tokens": max_length,
            "messages": messages
        }

        return anthropic_request_body


class MetaModel(AWSBedrockModel):

    def generate(
            self,
            ins: str,
            his: List[Tuple[str, str]] = [],
            max_length: int = 4096,
            top_p: float = 0.7,
            temperature: float = 0.9,
            **kwargs
    ):
        request_id = random_uuid() if "request_id" not in kwargs else kwargs["request_id"]

        fin_ins = self._generate_meta_instruction_from_history(ins, his)

        max_length = min(2048, max_length)

        meta_request_body = json.dumps({
            "prompt": fin_ins,
            "max_gen_len": max_length,
            "temperature": temperature,
            "top_p": top_p,
        })

        logger.info(f"Receiving request model: {self.model_id} messages: {meta_request_body}")

        answer = None
        input_tokens = 0
        output_tokens = 0
        time_taken = 0
        start_time = time.monotonic()

        try:
            api_res = self.bedrock.invoke_model(
                body=meta_request_body,
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            time_taken = time.monotonic() - start_time
            response_body = json.loads(api_res.get('body').read())
            logger.info(
                f"Completed request {request_id}: "
                f"model: {self.model_id} "
                f"cost: {time_taken} "
                f"result: {response_body}"
            )
            answer = response_body['generation']
            input_tokens = response_body['prompt_token_count']
            output_tokens = response_body['generation_token_count']
        except Exception as e:
            logger.error(f"request aws bedrock failed: {e}")
            answer = f"Exception occurred during the request, please try again: {e}" if not answer else answer

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

    def _generate_meta_instruction_from_history(
            self,
            ins: str,
            his: List[Dict[str, str]],
            role_mapping: Dict[str, str] = {
                "user": "Human",
                "assistant": "Assistant",
            }
    ):
        # if len(his) == 1 and ins == "":
        #     return his[0]['content'].strip() + "\n"

        new_his = []
        for item in his:
            if item["role"] == "system":
                new_his.append(item["content"])
                continue
            if item["role"] == "user":
                new_his.append(
                    f"[INST]{role_mapping[item['role']]}: {item['content']}")
                continue
            if item["role"] == "assistant":
                new_his.append(
                    f"{role_mapping[item['role']]}: {item['content']}[/INST]")
                continue

        # here we should make sure the user build the conversation string manually also
        # works. This means if the user do not provide  the history, then
        # we should treat ins as conversation string which the user build manually
        if len(new_his) > 0 and ins != "":
            new_his.append(f"[INST]{role_mapping['user']}: {ins}")
            new_his.append(f"{role_mapping['assistant']}:[/INST]")

        if len(new_his) > 0 and ins == "":
            new_his.append(f"{role_mapping['assistant']}[/INST]:")

        if len(new_his) == 0:
            new_his.append(ins)

        fin_ins = "\n".join(new_his)
        return fin_ins


class AwsBedrockModelFactory:
    @staticmethod
    def get_bedrock_model(
            model_provider: str,
            model_id: str,
            region_name: str = None,
            aws_access_key: str = None,
            aws_secret_key: str = None,
            model_api_version: Optional[str] = None,
    ) -> AWSBedrockModel:
        model_provider = model_provider.lower()
        logger.info(f"Using model_provider: {model_provider}")
        if SupportModelProviders.Anthropic == model_provider:
            return AnthropicModel(
                model_id=model_id,
                model_api_version=model_api_version,
                region_name=region_name,
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key
            )
        elif SupportModelProviders.Meta == model_provider:
            return MetaModel(
                model_id=model_id,
                model_api_version=model_api_version,
                region_name=region_name,
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key
            )
        else:
            raise ValueError(f"Unsupported model_provider: {model_provider}")
