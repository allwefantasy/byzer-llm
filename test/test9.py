from byzerllm.apps.client import ByzerLLMClient
from byzerllm.apps import ClientParams
import json

client = ByzerLLMClient(params=ClientParams(llm_chat_func="movice_qa",owner="william"))

q = "我想看一部三国的片子"

s = client.chat(f"请阅读上面的影视剧集信息，然后根据我的问题帮我找到合适的影视剧名称。 现在我的问题是： {q}，请只输出片名。",
                history=[],
                extra_query={"k":10,"prompt":"show_only_context"})

print(json.loads(s))
