import json
from typing import Dict, List, Union, Any
from tokenizers import Tokenizer


def get_meta(self):
    return [
        {
            "model_deploy_type": "proprietary",
            "backend": "tokenizers",
            "message_format": True,
        }
    ]


def process_input(ins: Union[str, List[Dict[str, Any]], Dict[str, Any]]):

    if isinstance(ins, list) or isinstance(ins, dict):
        return ins

    content = []
    try:
        ins_json = json.loads(ins)
    except:
        return ins

    ## 如果是字典，应该是非chat格式需求，比如语音转文字等
    if isinstance(ins_json, dict):
        return ins_json

    if isinstance(ins_json, list):
        if ins_json and isinstance(ins_json[0], dict):
            # 根据key值判断是什么类型的输入，比如语音转文字，语音合成等
            for temp in ["input", "voice", "audio", "audio_url"]:
                if temp in ins_json[0]:
                    return ins_json[0]

    content = []
    #     [
    #     {"type": "text", "text": "What’s in this image?"},
    #     {
    #       "type": "image_url",
    #       "image_url": {
    #         "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    #         "detail": "high"
    #       },
    #     },
    #   ],
    for item in ins_json:
        # for format like this: {"image": "xxxxxx", "text": "What’s in this image?","detail":"high"}
        # or {"image": "xxxxxx"}, {"text": "What’s in this image?"}
        if ("image" in item or "image_url" in item) and "type" not in item:
            image_data = item.get("image", item.get("image_url", ""))
            ## "data:image/jpeg;base64,"
            if not image_data.startswith("data:"):
                image_data = "data:image/jpeg;base64," + image_data

            ## get the other fields except image/text/image_url
            other_fields = {
                k: v for k, v in item.items() if k not in ["image", "text", "image_url"]
            }
            content.append(
                {
                    "image_url": {"url": image_data, **other_fields},
                    "type": "image_url",
                }
            )

        if "text" in item and "type" not in item:
            text_data = item["text"]
            content.append({"text": text_data, "type": "text"})

        ## for format like this: {"type": "text", "text": "What’s in this image?"},
        ## {"type": "image_url", "image_url": {"url":"","detail":"high"}}
        ## this is the standard format, just return it
        if "type" in item and item["type"] == "text":
            content.append(item)

        if "type" in item and item["type"] == "image_url":
            content.append(item)

    if not content:
        return ins

    return content

    @byzerllm.prompt()
    def extract_relevance_range_from_docs_with_conversation(
        self, conversations: List[Dict[str, str]], documents: List[str]
    ) -> str:
        """
        根据提供的文档和对话历史提取相关信息范围。

        输入:
        1. 文档内容:
        {% for doc in documents %}
        {{ doc }}
        {% endfor %}

        2. 对话历史:
        {% for msg in conversations %}
        <{{ msg.role }}>: {{ msg.content }}
        {% endfor %}

        任务:
        1. 分析最后一个用户问题及其上下文。
        2. 在文档中找出与问题相关的一个或多个重要信息段。
        3. 对每个相关信息段，确定其起始行号(start_line)和结束行号(end_line)。
        4. 信息段数量不超过4个。

        输出要求:
        1. 返回一个JSON数组，每个元素包含"start_line"和"end_line"。
        2. start_line和end_line必须是整数，表示文档中的行号。
        3. 行号从1开始计数。
        4. 如果没有相关信息，返回空数组[]。

        输出格式:
        严格的JSON数组，不包含其他文字或解释。

        示例:
        1.  文档：
            1 这是这篇动物科普文。
            2 大象是陆地上最大的动物之一。
            3 它们生活在非洲和亚洲。
            问题：大象生活在哪里？
            返回：[{"start_line": 2, "end_line": 3}]

        2.  文档：
            1 地球是太阳系第三行星，
            2 有海洋、沙漠，温度适宜，
            3 是已知唯一有生命的星球。
            4 太阳则是太阳系的唯一恒心。
            问题：地球的特点是什么？
            返回：[{"start_line": 1, "end_line": 3}]

        3.  文档：
            1 苹果富含维生素。
            2 香蕉含有大量钾元素。
            问题：橙子的特点是什么？
            返回：[]
        """


def stream_chat(
    self,
    tokenizer,
    ins: str,
    his: List[Dict[str, str]] = [],
    max_length: int = 4090,
    top_p: float = 0.95,
    temperature: float = 0.1,
    **kwargs,
):
    messages = [
        {"role": message["role"], "content": process_input(message["content"])}
        for message in his
    ] + [{"role": "user", "content": process_input(ins)}]

    encoded = self.encode(json.dumps(messages, ensure_ascii=False))
    return [
        (
            f"{len(encoded.tokens)}",
            {
                "metadata": {
                    "request_id": "",
                    "input_tokens_count": len(encoded.tokens),
                    "generated_tokens_count": 0,
                    "time_cost": 0,
                    "first_token_time": 0,
                    "speed": 0,
                }
            },
        )
    ]


def init_model(
    model_dir, infer_params: Dict[str, str] = {}, sys_conf: Dict[str, str] = {}
):

    model = Tokenizer.from_file(model_dir)
    import types

    model.stream_chat = types.MethodType(stream_chat, model)
    model.get_meta = types.MethodType(get_meta, model)

    return (model, None)
