import json
from typing import List, Tuple, Dict

import boto3


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.aws_access_key_id = infer_params["saas.aws_access_key_id"]
        self.aws_secret_access_key = infer_params["saas.aws_secret_access_key"]
        self.region_name = infer_params["saas.region_name"]
        self.model = infer_params["saas.model"]

        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )

    # saas/proprietary
    def get_meta(self):
        return [{
            "model_deploy_type": "saas",
            "backend": "saas",
        }]

    def stream_chat(
            self,
            tokenizer, ins: str,
            his: List[Tuple[str, str]] = [],
            max_length: int = 4096,
            top_p: float = 0.7,
            temperature: float = 0.9,
            **kwargs
    ):

        fin_ins = _generate_instruction_from_history(ins, his)

        max_length = min(2048, max_length)

        body = json.dumps({
            "prompt": fin_ins,
            "max_gen_len": max_length,
            "temperature": temperature,
            "top_p": top_p,
        })

        print(f"【Byzer --> AWS-Bedrock({self.model})】:\n{fin_ins}")

        response = None
        try:
            api_res = self.bedrock.invoke_model(
                body=body,
                modelId=self.model,
                accept='application/json',
                contentType='application/json'
            )
            response_body = json.loads(api_res.get('body').read())
            print(
                f"【AWS-Bedrock({self.model}) --> Byzer】:\n{json.dumps(response_body, ensure_ascii=False, indent=2)}"
            )
            response = response_body['generation']
        except Exception as e:
            print(f"request aws bedrock failed: {e}")
            response = f"Exception occurred during the request, please try again: {e}" \
                if response is None or response == "" else response

        return [(response, "")]


def _generate_instruction_from_history(
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
