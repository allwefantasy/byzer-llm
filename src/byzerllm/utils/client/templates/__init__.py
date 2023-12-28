LLM_TEMPALTE = '''你的目标是根据一个示例，帮我生成相关代码。

示例：

```text
<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n[/INST]</s><s>[INST] 你好 [/INST]
```

当我给定上述示例时，你需要生成以下代码：

```python
from langchain.prompts import PromptTemplate
from byzerllm.utils.client import Template   
from typing import Dict,Any,List,Optional,Union,Tuple,Callable

def tpl():
    def sys_format(t,v):
            m = PromptTemplate.from_template(t)
            return m.format(system_msg=v)
        
        def user_format(t,v):
            return f"<s>[INST] {v} [/INST]"
        
        def assistant_format(t,v):
            return f" {v} </s>"
        
        return Template(
            role_mapping={
               "user_role":"",
               "assistant_role": "",
               "system_msg":"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n[/INST]</s>",
               "system_msg_func":sys_format,
               "user_role_func": user_format,
               "assistant_role_func": assistant_format
            },            
            generation_config={},
            clean_func=lambda s: s
        )
```
当用户如如下对话时：

```python
[
  {"role","user","content":"你好"}
]
```

我们会使用如下规则来评估你的生成结果：

其中 role_mapping 是来自 tpl 方法中的 Template 对象中的 role_mapping 属性，该属性是一个字典，其中包含了用户、助手、系统的角色映射关系，你需要根据该映射关系来生成对话。

```
def generate_instruction_from_history(conversations:List[Dict[str,str]],role_mapping:Dict[str,str]={        
        "user_role":"User:",        
        "assistant_role":"Assistant:",
    }):
        
        new_his = []    
        for item in conversations:
            if item["role"] == "system":
                value = item["content"]
                if "system_msg_func" in role_mapping:
                    value = role_mapping["system_msg_func"](t=role_mapping["system_msg"],v=item["content"])
                new_his.append(value)
                continue
            
            if item["role"] == "user":
                value =  f"{role_mapping['user_role']}{item['content']}"
                if "user_role_func" in role_mapping:
                        value = role_mapping["user_role_func"](t=role_mapping["user_role"],v=item["content"])         
                new_his.append(value)  
            
            if item["role"] == "assistant":
                value =  f"{role_mapping['assistant_role']}{item['content']}"
                if "user_role_func" in role_mapping:
                        value = role_mapping["assistant_role_func"](t=role_mapping["assistant_role"],v=item["content"])         
                new_his.append(value)              
        
        if conversations[-1]["role"] == "user":            
            new_his.append(f"{role_mapping['assistant_role']}")

        fin_ins = "\n".join(new_his)
        return fin_ins
```

经过 `generate_instruction_from_history` 方法的运行，最后会输出如下结果：{example}

现在用户的模板是：{example} ,请生成代码。

务必注意： 请只生成 `tpl` 方法。
'''