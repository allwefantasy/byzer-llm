<p align="center">
  <picture>    
    <img alt="Byzer-LLM" src="https://github.com/allwefantasy/byzer-llm/blob/master/docs/source/assets/logos/logo.jpg" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap pretrain,finetune, serving for everyone
</h3>

<p align="center">
| <a href="#"><b>Documentation</b></a> | <a href="#"><b>Blog</b></a> | | <a href="#"><b>Discord</b></a> |

</p>

---

*Latest News* 🔥

- [2024/01] Release Byzer-LLM 0.1.33
- [2023/12] Release Byzer-LLM 0.1.30

---

Byzer-LLM is Ray based , a full lifecycle solution for LLM that includes pretrain, fintune, deployment and serving.

The unique features of Byzer-LLM are:

1. Full lifecyle: pretrain and finetune,deploy and serving support
2. Python/SQL API support
3. Ray based, easy to scale

---

* [Versions](#Versions)
* [Installation](#Installation)
* [Quick Start](#Quick-Start)
* [Embedding Model](#Embedding-Model)
* [Quatization](#Quatization)
* [Supported Models](#Supported-Models)
* [vLLM Support](#vLLM-Support)
* [DeepSpeed Support](#DeepSpeed-Support)
* [Function Calling](#Function-Calling)
* [Respond with pydantic class](#Respond-with-pydantic-class)
* [Function Implementation](#Function-Implementation)
* [LLM-Friendly Function/DataClass](#LLM-Friendly-Function/DataClass)
* [Model Meta](#Model-Meta)
* [Chat Template](#Chat-Template)
* [LLM Default Generation Parameters](#LLM-Default-Generation-Parameters)
* [SaaS Models](#SaaS-Models)
* [Multi Modal](#Multi-Modal)
* [StableDiffusion](#StableDiffusion)
* [SQL Support](#SQL-Support)
* [Pretrain](#Pretrain)
* [Finetune](#Finetune)
* [Stream Chat](#Stream-Chat)
* [Contributing](#Contributing)

---

## Versions
- 0.1.33： Fix Response Class bugs/ Add function implementation
- 0.1.32： StableDiffusion optimization
- 0.1.31： Stream Chat with token count information / Optimize multi modal model chat
- 0.1.30： Apply chat template for vLLM backend
- 0.1.29： Enhance DataAnalysis Agent
- 0.1.28： Bug fix
- 0.1.27： Bug fix
- 0.1.26： Support QianWen Saas/ Support stream chat in QianWenSaas/ Fix some Saas model bugs
- 0.1.24： Support get meta from model instance and auto setup template
- 0.1.23： Fintune with python API/ Fix some bugs
- 0.1.22： Function Calling support/ Response with pydantic class
- 0.1.19： Fix embedding bugs
- 0.1.18： Support stream chat/ Support Model Template
- 0.1.17： None
- 0.1.16： Enhance the API for byzer-retrieval
- 0.1.14： add get_tables/get_databases API for byzer-retrieval
- 0.1.13: support shutdown cluster for byzer-retrieval
- 0.1.12: Support Python API (alpha)
- 0.1.5:  Support python wrapper for [byzer-retrieval](https://github.com/allwefantasy/byzer-retrieval)

---

## Installation

```bash
pip install -r requirements.txt
pip install -U vllm
pip install -U byzerllm
ray start --head
```

---

## Quick Start

```python
import ray
from byzerllm.utils.client import ByzerLLM,LLMRequest,InferBackend

ray.init(address="auto",namespace="default",ignore_reinit_error=True)

llm = ByzerLLM()

llm.setup_gpus_per_worker(4).setup_num_workers(1)
llm.setup_infer_backend(InferBackend.transformers)

llm.deploy(model_path="/home/byzerllm/models/openbuddy-llama2-13b64k-v15",
           pretrained_model_type="custom/llama2",
           udf_name="llama2_chat",infer_params={})

llm.chat("llama2_chat",LLMRequest(instruction="hello world"))[0].output
```

The above code will deploy a llama2 model and then use the model to infer the input text. If you use transformers as the inference backend, you should specify the `pretrained_model_type` manually since the transformers backend can not auto detect the model type.

Byzer-LLM also support `deploy` SaaS model with the same way. This feature provide a unified interface for both open-source model and SaaS model. The following code will deploy a Azure OpenAI model and then use the model to infer the input text.


```python
import ray
from byzerllm.utils.client import ByzerLLM,LLMRequest,InferBackend
ray.init(address="auto",namespace="default",ignore_reinit_error=True)

llm = ByzerLLM()

llm.setup_gpus_per_worker(0).setup_num_workers(10)
llm.setup_infer_backend(InferBackend.transformers)

llm.deploy(pretrained_model_type="saas/azure_openai",
           udf_name="azure_openai",
           infer_params={
            "saas.api_type":"azure",
            "saas.api_key"="xxx"
            "saas.api_base"="xxx"
            "saas.api_version"="2023-07-01-preview"
            "saas.deployment_id"="xxxxxx"
           })

llm.chat("azure_openai",LLMRequest(instruction="hello world"))[0].output
```

Notice that the SaaS model does not need GPU, so we set the `setup_gpus_per_worker` to 0, and you can use `setup_num_workers`
to control max concurrency,how ever, the SaaS model has its own max concurrency limit, the `setup_num_workers` only control the max
concurrency accepted by the Byzer-LLM.

## Embedding Model

The following code is a example of deploying BGE embedding model

```python
import ray
from byzerllm.utils.client import ByzerLLM,LLMRequest,InferBackend
ray.init(address="auto",namespace="default",ignore_reinit_error=True)
llm = ByzerLLM()

llm.setup_gpus_per_worker(0.4).setup_num_workers(2).setup_infer_backend(InferBackend.Transformers)
llm.deploy(
    model_path="/home/byzerllm/models/bge-large-zh",
    pretrained_model_type="custom/bge",
    udf_name="emb",
    infer_params={}
)   
```

Then you can convert any text to vector :

```python
t = llm.emb("emb",LLMRequest(instruction="wow"))
t[0].output
#output: [-0.005588463973253965,
 -0.01747054047882557,
 -0.040633779019117355,
...
 -0.010880181565880775,
 -0.01713103987276554,
 0.017675869166851044,
 -0.010260719805955887,
 ...]
```

## Quatization

If the backend is `InferBackend.transformers`, here is the baichuan2 example:

```python
llm.setup_gpus_per_worker(2).setup_num_workers(1).setup_infer_backend(InferBackend.Transformers)
llm.deploy(
    model_path=model_location,
    pretrained_model_type="custom/baichuan2",
    udf_name="baichuan2_13_chat",
    infer_params={"quatization":"4"}
)
```
The available `quatization` values:

1. 4
2. 8
3. true/false

When it's set true, the int4 will be choosed.

If the bakcend is `InferBackend.VLLM`, here is the Yi example:

If you need to deploy model with Quantization, you can set the `infer_params` as the following code:

```python
llm.setup_gpus_per_worker(1).setup_num_workers(1).setup_infer_backend(InferBackend.VLLM)
llm.deploy(
    model_path="/home/winubuntu/models/Yi-6B-Chat-4bits",
    pretrained_model_type="custom/auto",
    udf_name="chat",
    infer_params={"backend.quantization":"AWQ"}
)
```

The parameter `backend.quantization` can be GPTQ/AWQ.


## Supported Models

The supported open-source `pretrained_model_type` are:

1. custom/llama2
2. bark	
3. whisper	
3. chatglm6b
4. custom/chatglm2
5. moss
6. custom/alpha_moss
7. dolly
8. falcon
9. llama
10. custom/starcode
11. custom/visualglm
12. custom/m3e
13. custom/baichuan
14. custom/bge
15. custom/qwen_vl_chat
16. custom/stable_diffusion
17. custom/zephyr

The supported SaaS `pretrained_model_type` are:

1. saas/chatglm	Chatglm130B
2. saas/sparkdesk	星火大模型
3. saas/baichuan	百川大模型
4. saas/zhipu	智谱大模型
5. saas/minimax	MiniMax 大模型
6. saas/qianfan	文心一言
7. saas/azure_openai	
8. saas/openai

Notice that the derived models from llama/llama2/startcode are also supported. For example, you can use `llama` to load vicuna model.

## vLLM Support

The Byzer-llm also support vLLM as the inference backend. The following code will deploy a vLLM model and then use the model to infer the input text.

```python
import ray
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.client import ByzerLLM,LLMRequest,InferBackend

ray.init(address="auto",namespace="default",ignore_reinit_error=True)
llm = ByzerLLM()

llm.setup_gpus_per_worker(2)
llm.setup_num_workers(1)
llm.setup_infer_backend(InferBackend.VLLM)

llm.deploy(
    model_path="/home/byzerllm/models/openbuddy-zephyr-7b-v14.1",
    pretrained_model_type="custom/auto",
    udf_name="zephyr_chat"",
    infer_params={}
)

llm.chat("zephyr_chat",LLMRequest(instruction="hello world"))[0].output
```

There are some tiny differences between the vLLM and the transformers backend. 

1. The `pretrained_model_type` is fixed to `custom/auto` for vLLM, since the vLLM will auto detect the model type.
2. Use `setup_infer_backend` to specify `InferBackend.VLLM` as the inference backend.
 

### Stream Chat

If the model deployed with the backend vLLM, then it also support `stream chat`：
the `stream_chat_oai` will return a generator, you can use the generator to get the output text.

```python

llm.setup_default_model_name(chat_model_name) 

t = llm.stream_chat_oai(conversations=[{
    "role":"user",
    "content":"Hello, how are you?"
}])

for line in t:
   print(line+"\n")
```

## DeepSpeed Support

The Byzer-llm also support DeepSpeed as the inference backend. The following code will deploy a DeepSpeed model and then use the model to infer the input text.

```python
import ray
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.client import ByzerLLM,LLMRequest,InferBackend

ray.init(address="auto",namespace="default",ignore_reinit_error=True)
llm = ByzerLLM()

llm.setup_gpus_per_worker(4)
llm.setup_num_workers(1)
llm.setup_infer_backend(InferBackend.DeepSpeed)

llm.deploy(
    model_path="/home/byzerllm/models/openbuddy-llama-13b-v5-fp16",
    pretrained_model_type="custom/auto",
    udf_name="llama_chat"",
    infer_params={}
)

llm.chat("llama_chat",LLMRequest(instruction="hello world"))[0].output
```

The code above is totally the same as the code for vLLM, except that the `InferBackend` is `InferBackend.DeepSpeed`.


## Function Calling

Here is a simple example for function calling based on QWen 72B

Deploy Model:


```python
import ray
ray.init(address="auto",namespace="default") 
llm = ByzerLLM()

model_location="/home/byzerllm/models/Qwen-72B-Chat"

llm.setup_gpus_per_worker(8).setup_num_workers(1).setup_infer_backend(InferBackend.VLLM)
llm.deploy(
    model_path=model_location,
    pretrained_model_type="custom/auto",
    udf_name=chat_model_name,
    infer_params={}
)

llm.setup_default_model_name("chat")
# from 0.1.24 
# llm.setup_auto("chat")
meta = llm.get_meta()
llm.setup_max_model_length("chat",meta.get("max_model_len",32000))
lm.setup_template("chat",Templates.qwen()) 
```

Try to create some Python functions:

```python

from typing import List,Dict,Any,Annotated
import pydantic 
import datetime
from dateutil.relativedelta import relativedelta

def compute_date_range(count:Annotated[int,"时间跨度，数值类型"],
                       unit:Annotated[str,"时间单位，字符串类型",{"enum":["day","week","month","year"]}])->List[str]:
    '''
    计算日期范围

    Args:
        count: 时间跨度，数值类型
        unit: 时间单位，字符串类型，可选值为 day,week,month,year
    '''        
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    if unit == "day":
        return [(now - relativedelta(days=count)).strftime("%Y-%m-%d %H:%M:%S"),now_str]
    elif unit == "week":
        return [(now - relativedelta(weeks=count)).strftime("%Y-%m-%d %H:%M:%S"),now_str]
    elif unit == "month":
        return [(now - relativedelta(months=count)).strftime("%Y-%m-%d %H:%M:%S"),now_str]
    elif unit == "year":
        return [(now - relativedelta(years=count)).strftime("%Y-%m-%d %H:%M:%S"),now_str]
    return ["",""]

def compute_now()->str:
    '''
    计算当前时间
    '''
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

Here we provide two functions:

1. compute_date_range: compute the date range based on the count and unit
2. compute_now: get the current date

We will use the model to call these tools according to the user's question.

```python
t = llm.chat_oai([{
    "content":'''计算当前时间''',
    "role":"user"    
}],tools=[compute_date_range,compute_now],execute_tool=True)

t[0].values

## output: ['2023-12-18 17:30:49']
```

```python
t = llm.chat_oai([{
    "content":'''最近三个月趋势''',
    "role":"user"    
}],tools=[compute_date_range,compute_now],execute_tool=True)

t[0].values

## output: [['2023-09-18 17:31:21', '2023-12-18 17:31:21']]
```

```python
t = llm.chat_oai([{
    "content":'''最近三天''',
    "role":"user"    
}],tools=[compute_date_range,compute_now],execute_tool=True)

t[0].values

## output: [['2023-12-15 17:23:38', '2023-12-18 17:23:38']]
```

```python
t = llm.chat_oai([{
    "content":'''你吃饭了么？''',
    "role":"user"    
}],tools=[compute_date_range,compute_now],execute_tool=True)

if t[0].values:
    print(t[0].values[0])
else:
    print(t[0].response.output)   

## output: '您好，我是一个人工智能语言模型，暂时无法吃饭。'
```

You can check the default prompt template function in `from byzerllm.utils import function_calling_format`.
If the model is not work well with the default function, you can setup your custom function:

```python
def custom_function_calling_format(prompt:str,tools:List[Callable],tool_choice:Callable)->str:
.....


llm.setup_function_calling_format_func("chat",custom_function_calling_format)
```

## Respond with pydantic class

When you chat with LLM, you can specify the reponse class, 

```python
import pydantic 

class Story(pydantic.BaseModel):
    '''
    故事
    '''

    title: str = pydantic.Field(description="故事的标题")
    body: str = pydantic.Field(description="故事主体")



t = llm.chat_oai([
{
    "content":f'''请给我讲个故事，分成两个部分，一个标题，一个故事主体''',
    "role":"user"
},
],response_class=Story)

t[0].value

## output: Story(title='勇敢的小兔子', body='在一个美丽的森林里，住着一只可爱的小兔子。小兔子非常勇敢，有一天，森林里的动物们都被大灰狼吓坏了。只有小兔子站出来，用智慧和勇气打败了大灰狼，保护了所有的动物。从此，小兔子成为了森林里的英雄。')
```

The above code will ask the LLM to generate the Story class directly. However, sometimes we hope the LLM 
generate text first, then extract the structure from the text, you can set `response_after_chat=True` to 
enable this behavior. However, this will bring some performance penalty(additional inference).

```python
t = llm.chat_oai([
{
    "content":f'''请给我讲个故事，分成两个部分，一个标题，一个故事主体''',
    "role":"user"
},
],response_class=Story,response_after_chat=True)

t[0].value
## output: Story(title='月光下的守护者', body='在一个遥远的古老村庄里，住着一位名叫阿明的年轻人。阿明是个孤儿，从小在村里长大，以种田为生。他善良、勤劳，深受村民们喜爱。\n\n村子里有个传说，每当满月时分，月亮女神会在村子后山的古树下出现，赐福给那些善良的人们。然而，只有最纯洁的心才能看到她。因此，每年的这个时候，阿明都会独自一人前往后山，希望能得到女神的祝福。\n\n这一年，村子遭受了严重的旱灾，庄稼枯黄，人们生活困苦。阿明决定向月亮女神祈求降雨，拯救村子。他在月光下虔诚地祈祷，希望女神能听到他的呼唤。\n\n就在这个时刻，月亮女神出现了。她被阿明的善良和执着所感动，答应了他的请求。第二天早晨，天空乌云密布，大雨倾盆而下，久旱的土地得到了滋润，庄稼重新焕发生机。\n\n从此以后，每年的满月之夜，阿明都会去后山等待月亮女神的出现，他成为了村民心中的守护者，用他的善良和执着，守护着整个村庄。而他也终于明白，真正的守护者，并非需要超凡的力量，只需要一颗充满爱与善良的心。')
```

You can check the default prompt template function in `from byzerllm.utils import response_class_format,response_class_format_after_chat`.
If the model is not work well with the default function, you can setup your custom function:

```python
def custom_response_class_format(prompt:str,cls:pydantic.BaseModel)->str:
.....


llm.setup_response_class_format_func("chat",custom_response_class_format)
```

## Function Implementation

The Byzer-llm also support function implementation. You can define a empty function, and combine the doc in the function/the user's quesion to guide the LLM to implement the function. 

Here is a simple example:

```python
class TimeRange(pydantic.BaseModel):
    '''
    时间区间
    格式需要如下： yyyy-MM-dd
    '''  
    
    start: str = pydantic.Field(...,description="开始时间.时间格式为 yyyy-MM-dd")
    end: str = pydantic.Field(...,description="截止时间.时间格式为 yyyy-MM-dd")


def calculate_time_range():
    '''
    计算时间区间，时间格式为 yyyy-MM-dd. 
    '''
    pass 
    
t = llm.chat_oai([{
    "content":"去年三月到七月",
    "role":"user"    
}],impl_func=calculate_time_range,response_class=TimeRange,execute_impl_func=True)
```

The above code , we define a function called `calculate_time_range`, and the function is empty, then we discribe the function in the doc string, and define the response class `TimeRange`, to make sure the return value is a `TimeRange` instance. Since the function should be used to resolve the user's question, so the implementation of the function should be related to the user's question. Instead try to implement a common use function, we can just implement a function which can only resolve the user's current question.

After the execution, you can get the output like this:

```python
t[0].value
# start='2023-03-01' end='2023-07-31'
```

If the value is None or not correct, you can get the error message:

```python
t[0].metadata.get("resason","")
```

If your function has parameters, you can pass the parameters to the function by `impl_func_params`:

```python
t = llm.chat_oai([{
    "content":"xxxxx",
    "role":"user"    
}],
impl_func=calculate_time_range,
impl_func_params={},
response_class=TimeRange,execute_impl_func=True)
```

If you want to replace the default prompt template function, here is a example:

```python
import pydantic
from typing import List,Optional,Union,Callable
from byzerllm.utils import serialize_function_to_json

def function_impl_format2(prompt:str,func:Optional[Union[Callable,str]],
                             cls:Union[pydantic.BaseModel,str])->str:
    
    tool_choice_ser = serialize_function_to_json(func)    
    _cls = ""
    if isinstance(cls, str):
        _cls = cls
    else:
        _cls = cls.schema_json(ensure_ascii=False)
    
    msg = f''''生成一个python函数，给出详细的思考逻辑，对最后生成的函数不要进行示例说明。

生成的函数的名字以及参数需要满足如下约束：

\```json
{tool_choice_ser}
\```

生成的函数的返回值必须是 Json 格式。

下面是使用 OpenAPI 3.1. 规范描述了你需如何进行json格式的生成。

\```json
{_cls}
\```

根据用的户问题,{func.__doc__}。用户的问题是：{prompt}

请你实现这个函数。
''' 
    
    return msg

llm.setup_impl_func_format_func(chat_model_name,function_impl_format2)
```

The default prompt template function is `function_impl_format`, you can check the source code in `from byzerllm.utils import function_impl_format`.


## LLM-Friendly Function/DataClass

If you want to improve the performance of Function Calling or Response Class, you should make your Function(Tool) and Data Class is LLM-Friendly.  

Let's take a look at the following python code:

```python
def compute_date_range(count:int, unit:str)->List[str]:                   
    now = datetime.datetime.now()
    ....
```

This code is not LLM-Friendly Function since it's difficult to know the usage of this funciton and 
what's the meaning of the input parameters.

The LLM just like human, it's hard to let the LLM know when or how to invoke this function. Especially the parameter `unit`
actually is enum value but the LLM no way to get this message.

So, in order to make the LLM knows more about this function in Byzer-LLM, you should 
follow some requirments:

1. Adding pythonic function comment 
2. Use annotated to provide type and comment for every parameter, if the parameter is a enum, then provide enum values.

Here is the LLM-Friendly fuction definision.

```python
def compute_date_range(count:Annotated[int,"时间跨度，数值类型"],
                       unit:Annotated[str,"时间单位，字符串类型",{"enum":["day","week","month","year"]}])->List[str]:
    '''
    计算日期范围

    Args:
        count: 时间跨度，数值类型
        unit: 时间单位，字符串类型，可选值为 day,week,month,year
    '''        
    now = datetime.datetime.now()
    ....
```

If the LLM make something wrong to your function (e.g. provide the bad parameters), try to optimize the function comment 
and the parameter Annotated comment.

## Model Meta

The Byzer-llm also support get the model meta information. The following code will get the meta information of model instance called `chat`:

```python

```python
llm.get_meta(model="chat")

#output:
# {'model_deploy_type': 'proprietary',
#  'backend': 'ray/vllm',
#  'max_model_len': 32768,
#  'architectures': ['QWenLMHeadModel']}
```

## Chat Template

The different models have different chat templates, the Byzer-LLM have provide some chat templates for the models. You can use the following code to setup the chat template:

```python
from byzerllm.utils.client import Templates
llm.setup_template("chat",Templates.qwen()) 
```

However, we also support the `tokeninzer.apply_chat_template`, you can use the following code to apply the chat template:

```python
llm.setup_template("chat","auto") 
```

If the model is not work well with the `tokeninzer.apply_chat_template`, this function will raise an exception. In this case, you can use the `llm.setup_template` to setup the chat template manually.

You can also use the `llm.get_meta` to check if the model support the `apply_chat_template`:

```python
llm.get_meta(model="chat")
```

The output:

```json
{'model_deploy_type': 'proprietary',
 'backend': 'ray/vllm',
 'support_stream': True,
 'support_chat_template': True,
 'max_model_len': 4096,
 'architectures': ['LlamaForCausalLM']}
```

Notice that this feature will cause additional RPC call, so it will bring some performance penalty.

## LLM Default Generation Parameters

The Byzer-llm also support setup the default generation parameters for the model. The following code will setup the default generation parameters for the model instance called `chat`:

```python
llm.setup_extra_generation_params("chat",{
    "generation.stop_token_ids":[7]
})
```

In this case, the `generation.stop_token_ids` will be set to `[7]` for the model instance `chat`. Every time you call the `chat` model, the `generation.stop_token_ids` will be set to `[7]` automatically.

## Multi Modal 

The Byzer-llm also support multi modal. The following code will deploy a multi modal model and then use the model to infer the input text.

```python
import ray
from byzerllm.utils.client import ByzerLLM,InferBackend

ray.init(address="auto",namespace="default")   

llm = ByzerLLM()
chat_model_name = "qwen_vl_chat"
model_location = "/home/byzerllm/models/Qwen-VL-Chat"

llm.setup_gpus_per_worker(1).setup_num_workers(1).setup_infer_backend(InferBackend.Transformers)
llm.deploy(
    model_path=model_location,
    pretrained_model_type="custom/qwen_vl_chat",
    udf_name=chat_model_name,
    infer_params={}
)    
```

Then you can use the model to chat:

```python
import base64
image_path = "/home/byzerllm/projects/jupyter-workspace/1.jpg"
with open(image_path, "rb") as f:
    image_content = base64.b64encode(f.read()).decode("utf-8")

t = llm.chat_oai(conversations=[{
    "role": "user",
    "content": "这是什么"
}],model=chat_model_name,llm_config={"image":image_content})

t[0].output

# '{"response": "图中是一名女子在沙滩上和狗玩耍，旁边的狗是一只拉布拉多犬，它坐在沙滩上，面对着一名身穿格子衬衫的女子。女子的腿有些残疾，但是她依然坚持坐在沙滩上和狗玩耍。她的右手拿着一个小玩具，这个玩具上面有两行黑色字母，具体是什么内容看不清楚。她打算把玩具扔给拉布拉多犬。", "history": [{"role": "user", "content": "Picture 1: <img>/tmp/byzerllm/visualglm/images/23eb4cea-cb6e-4f55-8adf-3179ca92ab42.jpg</img>\\n这是什么"}, {"role": "assistant", "content": "图中是一名女子在沙滩上和狗玩耍，旁边的狗是一只拉布拉多犬，它坐在沙滩上，面对着一名身穿格子衬衫的女子。女子的腿有些残疾，但是她依然坚持坐在沙滩上和狗玩耍。她的右手拿着一个小玩具，这个玩具上面有两行黑色字母，具体是什么内容看不清楚。她打算把玩具扔给拉布拉多犬。"}]}'
```

You can chat multi rounds with the following code:

```python
import json
history = json.loads(t[0].output)["history"]

llm.chat_oai(conversations=history+[{
    "role": "user",
    "content": "能圈出狗么？"
}],model=chat_model_name,llm_config={"image":image_content})

# [LLMResponse(output='{"response": "<ref>狗</ref><box>(221,425),(511,889)</box>", "history": [{"role"
```

Get the history from the previous chat, then add the hisotry to new conversation, then chat again.

## StableDiffusion

The Byzer-llm also support StableDiffusion as the inference backend. The following code will deploy a StableDiffusion model and then use the model to infer the input text.

```python
import ray
from byzerllm.utils.client import ByzerLLM,InferBackend

ray.init(address="auto",namespace="default")   

llm = ByzerLLM()
chat_model_name = "sd_chat"
model_location = "/home/byzerllm/models/stable-diffusion-v1-5"

llm.setup_gpus_per_worker(2).setup_num_workers(1).setup_infer_backend(InferBackend.Transformers)
llm.deploy(
    model_path=model_location,
    pretrained_model_type="custom/stable_diffusion",
    udf_name=chat_model_name,
    infer_params={}
)

def show_image(content):
    from IPython.display import display, Image
    import base64             
    img = Image(base64.b64decode(content))
    display(img)    
    
```

Then you can chat with the model:

```python
import json
t = llm.chat_oai(
    conversations=[
        {
            "role":"user",
            "content":"画一只猫"
        }
    ],model=chat_model_name,llm_config={"gen.batch_size":3}
)

cats = json.loads(t[0].output)
for res in cats:
    show_image(res["img64"])
```

The output:

![](./images/cat2.png)

The parameters:

| 参数                        | 含义                                                         | 默认值   |
| --------------------------- | ------------------------------------------------------------ | -------- |
| Instruction                 | prompt                                                       | 非空     |
| generation.negative_prompt  | 反向的prompt                                                 | ""       |
| generation.sampler_name     | 调度名(unpic, euler_a,euler,ddim,ddpm,deis,dpm2,dpm2-a,dpm++_2m,dpm++_2m_karras,heun,heun_karras,lms,pndm:w) | euler_a  |
| generation.sampling_steps   | 生成的步骤数                                                 | 25       |
| generation.batch_size       | 一次生成几张                                                 | 1        |
| generation.batch_count      | 生成几次                                                     | 1        |
| generation.cfg_scale        | 随机或贴合程度值,值越小生成的图片离你的Tags描述的内容差距越大 | 7.5      |
| generation.seed             | 随机种子                                                     | -1       |
| generation.width            | 图片宽度                                                     | 768      |
| generation.height           | 图片高度                                                     | 768      |
| generation.enable_hires     | 开启高分辨率修复功能(和下面两个一组)                         | false    |
| generation.upscaler_mode    | 放大算法(bilinear, bilinear-antialiased,bicubic,bicubic-antialiased,nearest,nearest-exact) | bilinear |
| generation.scale_slider     | 放大比例                                                     | 1.5      |
| generation.enable_multidiff | 图片分割处理(减少显存销耗)(和下面3个一组)                    | false    |
| generation.views_batch_size | 分批处理规模                                                 | 4        |
| generation.window_size      | 切割大小，宽，高                                             | 64       |
| generation.stride           | 步长                                                         | 16       |
| generation.init_image       | 初始化图片，基于这个图片处理(必须传输base64加密的图片) (和下面的一组) | None     |
| generation.strength         | 重绘幅度: 图像模仿自由度，越高越自由发挥，越低和参考图像越接近，通常小于0.3基本就是加滤镜 | 0.5      |



## SQL Support

In addition to the Python API, Byzer-llm also support SQL API. In order to use the SQL API, you should install Byzer-SQL language first.

Try to install the Byzer-SQL language with the following command:

```bash
git clone https://gitee.com/allwefantasy/byzer-llm
cd byzer-llm/setup-machine
sudo -i 
ROLE=master ./setup-machine.sh
```

After the installation, you can visit the Byzer Console at http://localhost:9002. 

In the Byzer Console, you can run the following SQL to deploy a llama2 model which have the same effect as the Python code above.

```sql
!byzerllm setup single;
!byzerllm setup "num_gpus=4";
!byzerllm setup "maxConcurrency=1";
!byzerllm setup "infer_backend=transformers";

run command as LLM.`` where 
action="infer"
and pretrainedModelType="custom/llama2"
and localModelDir="/home/byzerllm/models/openbuddy-llama-13b-v5-fp16"
and reconnect="false"
and udfName="llama2_chat"
and modelTable="command";

```

Then you can invoke the model with UDF `llama2_chat`:

```sql

select 
llama2_chat(llm_param(map(
              "user_role","User",
              "assistant_role","Assistant",
              "system_msg",'You are a helpful assistant. Think it over and answer the user question correctly.',
              "instruction",llm_prompt('
Please remenber my name: {0}              
',array("Zhu William"))

))) as q 
as q1;
```

Once you deploy the model with `run command as LLM`, then you can ues the model as a SQL function. This feature is very useful for data scientists who want to use LLM in their data analysis or data engineers who want to use LLM in their data pipeline.

---

### QWen

If you use QWen in ByzerLLM, you should sepcify the following parameters mannualy:

1. the role mapping 
2. the stop_token_ids
3. trim the stop tokens from the output

However, we provide a template for this, try to the following code:

```python
from byzerllm.utils.client import Templates

### Here,we setup the template for qwen
llm.setup_template("chat",Templates.qwen())

t = llm.chat_oai(conversations=[{
    "role":"user",
    "content":"你好,给我讲个100字的笑话吧?"
}])
print(t)
```

---
## SaaS Models

Since the different SaaS models have different parameters, here we provide some templates for the SaaS models to help you deploy the SaaS models.

Here is quick example with Python:

```python

import ray
from byzerllm.utils.client import ByzerLLM

ray.init(address="auto",namespace="default",ignore_reinit_error=True)   

llm = ByzerLLM(verbose=True)

llm.setup_num_workers(1).setup_gpus_per_worker(0)

chat_name = "baichuan_chat2"
if llm.is_model_exist(chat_name):
    llm.undeploy(udf_name=chat_name)

llm.deploy(model_path="",
           pretrained_model_type="saas/baichuan",
           udf_name=chat_name,
           infer_params={
            "saas.api_key":"xxxxxxx",            
            "saas.baichuan_api_url":"https://api.baichuan-ai.com/v1/chat/completions",
            "saas.model":"Baichuan2-Turbo"
           })

llm.chat_oai(model=chat_name,conversations=[{
    "role":"user",
    "content":"你好",
}])           
```

Since we use SaaS model, There is no need to specify the gpus, so we set `setup_gpus_per_worker` to 0. However, the SaaS model has its own max concurrency limit, the `setup_num_workers` only control the max concurrency accepted by the Byzer-LLM.


For now, only QianWen Saas support stream chat, here is the example:

```python
from byzerllm.utils.client import ByzerLLM
llm = ByzerLLM(verbose=True)

llm.setup_num_workers(1).setup_gpus_per_worker(0)

chat_name = "qianwen_chat"
if llm.is_model_exist(chat_name):
    llm.undeploy(udf_name=chat_name)

llm.deploy(model_path="",
           pretrained_model_type="saas/qianwen",
           udf_name=chat_name,
           infer_params={
            "saas.api_key":"xxxxxxx",            
            "saas.model":"qwen-turbo"
           })

## here you can use `stream_chat_oai`
v = llm.stream_chat_oai(model=chat_name,conversations=[{
    "role":"user",
    "content":"你好，你是谁",
}],llm_config={"gen.incremental_output":False})

for t in v:
    print(t,flush=True)           
```

To check if a model the support of stream chat, you can check [Model Meta](Model-Meta).


### qianfan


```sql
!byzerllm setup single;
!byzerllm setup "num_gpus=0";
!byzerllm setup "maxConcurrency=10";

run command as LLM.`` where
action="infer"
and pretrainedModelType="saas/qianfan"
and `saas.api_key`="xxxxxxxxxxxxxxxxxx"
and `saas.secret_key`="xxxxxxxxxxxxxxxx"
and `saas.model`="ERNIE-Bot-turbo"
and `saas.retry_count`="3"
and `saas.request_timeout`="120"
and reconnect="false"
and udfName="qianfan_saas"
and modelTable="command";

```

### azure openai

```sql

!byzerllm setup single;
!byzerllm setup "num_gpus=0";
!byzerllm setup "maxConcurrency=10";

run command as LLM.`` where
action="infer"
and pretrainedModelType="saas/azure_openai"
and `saas.api_type`="azure"
and `saas.api_key`="xxx"
and `saas.api_base`="xxx"
and `saas.api_version`="2023-07-01-preview"
and `saas.deployment_id`="xxxxx"
and udfName="azure_openai"
and modelTable="command";
```

### openai

```sql

!byzerllm setup single;
!byzerllm setup "num_gpus=0";
!byzerllm setup "maxConcurrency=10";

run command as LLM.`` where
action="infer"
and pretrainedModelType="saas/azure_openai"
and `saas.api_type`="azure"
and `saas.api_key`="xxx"
and `saas.api_base`="xxx"
and `saas.api_version`="xxxxx"
and `saas.model`="xxxxx"
and udfName="openai_saas"
and modelTable="command";
```

### zhipu

```sql

!byzerllm setup single;
!byzerllm setup "num_gpus=0";
!byzerllm setup "maxConcurrency=10";

run command as LLM.`` where
action="infer"
and pretrainedModelType="saas/zhipu"
and `saas.api_key`="xxxxxxxxxxxxxxxxxx"
and `saas.secret_key`="xxxxxxxxxxxxxxxx"
and `saas.model`="chatglm_lite"
and udfName="zhipu_saas"
and modelTable="command";
```

### minimax

```sql

!byzerllm setup single;
!byzerllm setup "num_gpus=0";
!byzerllm setup "maxConcurrency=10";

run command as LLM.`` where
action="infer"
and pretrainedModelType="saas/minimax"
and `saas.api_key`="xxxxxxxxxxxxxxxxxx"
and `saas.group_id`="xxxxxxxxxxxxxxxx"
and `saas.model`="abab5.5-chat"
and `saas.api_url`="https://api.minimax.chat/v1/text/chatcompletion_pro"
and udfName="minimax_saas"
and modelTable="command";

```

### sparkdesk

```sql
!byzerllm setup single;
!byzerllm setup "num_gpus=0";
!byzerllm setup "maxConcurrency=10";

run command as LLM.`` where
action="infer"
and pretrainedModelType="saas/sparkdesk"
and `saas.appid`="xxxxxxxxxxxxxxxxxx"
and `saas.api_key`="xxxxxxxxxxxxxxxx"
and `saas.api_secret`="xxxx"
and `gpt_url`="ws://spark-api.xf-yun.com/v1.1/chat"
and udfName="sparkdesk_saas"
and modelTable="command";
```

### baichuan

```sql
!byzerllm setup single;
!byzerllm setup "num_gpus=0";
!byzerllm setup "maxConcurrency=10";

run command as LLM.`` where
action="infer"
and pretrainedModelType="saas/baichuan"
and `saas.api_key`="xxxxxxxxxxxxxxxxxx"
and `saas.secret_key`="xxxxxxxxxxxxxxxx"
and `saas.baichuan_api_url`="https://api.baichuan-ai.com/v1/chat"
and `saas.model`="Baichuan2-53B"
and udfName="baichuan_saas"
and modelTable="command";
```

---

## Pretrain

This section will introduce how to pretrain a LLM model with Byzer-llm.  However, for now, the pretrain feature is more mature in Byzer-SQL, so we will introduce the pretrain feature in Byzer-SQL.

```sql
-- Deepspeed Config
set ds_config='''
{
  "gradient_accumulation_steps": 1,
  "train_micro_batch_size_per_gpu": 1,
  "prescale_gradients": false,
  "zero_allow_untested_optimizer": true,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-8,
      "eps": 1.0e-8,
      "betas": [
        0.9,
        0.95
      ],
      "weight_decay": 0.1
    }
  },
  "tensorboard": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
         "device": "cpu"         
     },           
    "offload_param": {
         "device": "cpu"
    },
    "contiguous_gradients": true,
    "allgather_bucket_size": 1e8,
    "reduce_bucket_size": 1e8,
    "overlap_comm": true,
    "reduce_scatter": true
  },
  "steps_per_print": 16,
  "gradient_clipping": 1.0,
  "wall_clock_breakdown": true,
  "bf16": {
    "enabled": true
  }
}
''';

-- load data
load text.`file:///home/byzerllm/data/raw_data/*`
where wholetext="true" as trainData;

select value as text,file from trainData  as newTrainData;

-- split the data into 12 partitions
run newTrainData as TableRepartition.`` where partitionNum="12" and partitionCols="file" 
as finalTrainData;


-- setup env, we use 12 gpus to pretrain the model
!byzerllm setup sfft;
!byzerllm setup "num_gpus=12";

-- specify the pretrain model type and the pretrained model path
run command as LLM.`` where 
and localPathPrefix="/home/byzerllm/models/sfft/jobs"
and pretrainedModelType="sfft/llama2"
-- original model is from
and localModelDir="/home/byzerllm/models/Llama-2-7b-chat-hf"
-- and localDataDir="/home/byzerllm/data/raw_data"

-- we use async mode to pretrain the model, since the pretrain process will take several days or weeks
-- Ray Dashboard will show the tensorboard address, and then you can monitor the loss
and detached="true"
and keepPartitionNum="true"

-- use deepspeed config, this is optional
and deepspeedConfig='''${ds_config}'''


-- the pretrain data is from finalTrainData table
and inputTable="finalTrainData"
and outputTable="llama2_cn"
and model="command"
-- some hyper parameters
and `sfft.int.max_length`="128"
and `sfft.bool.setup_nccl_socket_ifname_by_ip`="true"
;
```

Since the deepspeed checkpoint is not compatible with the huggingface checkpoint, we need to convert the deepspeed checkpoint to the huggingface checkpoint. The following code will convert the deepspeed checkpoint to the huggingface checkpoint.

```sql
!byzerllm setup single;

run command as LLM.`` where 
action="convert"
and pretrainedModelType="deepspeed/llama3b"
and modelNameOrPath="/home/byzerllm/models/base_model"
and checkpointDir="/home/byzerllm/data/checkpoints"
and tag="Epoch-1"
and savePath="/home/byzerllm/models/my_3b_test2";
```


Now you can deploy the converted model :

```sql
-- 部署hugginface 模型
!byzerllm setup single;

set node="master";
!byzerllm setup "num_gpus=2";
!byzerllm setup "workerMaxConcurrency=1";

run command as LLM.`` where 
action="infer"
and pretrainedModelType="custom/auto"
and localModelDir="/home/byzerllm/models/my_3b_test2"
and reconnect="false"
and udfName="my_3b_chat"
and modelTable="command";
```

## Finetune

```sql
-- load data, we use the dummy data for finetune
-- data format supported by Byzer-SQL：https://docs.byzer.org/#/byzer-lang/zh-cn/byzer-llm/model-sft

load json.`/tmp/upload/dummy_data.jsonl` where
inferSchema="true"
as sft_data;

-- Fintune Llama2
!byzerllm setup sft;
!byzerllm setup "num_gpus=4";

run command as LLM.`` where 
and localPathPrefix="/home/byzerllm/models/sft/jobs"

-- 指定模型类型
and pretrainedModelType="sft/llama2"

-- 指定模型
and localModelDir="/home/byzerllm/models/Llama-2-7b-chat-hf"
and model="command"

-- 指定微调数据表
and inputTable="sft_data"

-- 输出新模型表
and outputTable="llama2_300"

-- 微调参数
and  detached="true"
and `sft.int.max_seq_length`="512";
```

You can check the finetune actor in the Ray Dashboard, the name of the actor is `sft-william-xxxxx`.

After the finetune actor is finished, you can get the model path, so you can deploy the finetuned model.


Here is the log of the finetune actor:

```
Loading data: /home/byzerllm/projects/sft/jobs/sft-william-20230809-13-04-48-674fd1b9-2fc1-45b9-9d75-7abf07cb84cb/finetune_data/data.jsonl3
2
there are 33 data in dataset
*** starting training ***
{'train_runtime': 19.0203, 'train_samples_per_second': 1.735, 'train_steps_per_second': 0.105, 'train_loss': 3.0778136253356934, 'epoch': 0.97}35

***** train metrics *****36  
epoch                    =       0.9737  
train_loss               =     3.077838  
train_runtime            = 0:00:19.0239  
train_samples_per_second =      1.73540  
train_steps_per_second   =      0.10541

[sft-william] Copy /home/byzerllm/models/Llama-2-7b-chat-hf to /home/byzerllm/projects/sft/jobs/sft-william-20230809-13-04-48-674fd1b9-2fc1-45b9-9d75-7abf07cb84cb/finetune_model/final/pretrained_model4243              
[sft-william] Train Actor is already finished. You can check the model in: /home/byzerllm/projects/sft/jobs/sft-william-20230809-13-04-48-674fd1b9-2fc1-45b9-9d75-7abf07cb84cb/finetune_model/final   
```

You can download the finetuned model from the path `/home/byzerllm/projects/sft/jobs/sft-william-20230809-13-04-48-674fd1b9-2fc1-45b9-9d75-7abf07cb84cb/finetune_model/final`, or copy the model to all other node in the Ray cluster.

Try to deploy the finetuned model:

```sql
!byzerllm setup single;
run command as LLM.`` where 
action="infer"
and localPathPrefix="/home/byzerllm/models/infer/jobs"
and localModelDir="/home/byzerllm/models/sft/jobs/sft-william-llama2-alpaca-data-ccb8fb55-382c-49fb-af04-5cbb3966c4e6/finetune_model/final"
and pretrainedModelType="custom/llama2"
and udfName="fintune_llama2_chat"
and modelTable="command";
```

Byzer-LLM use QLora to finetune the model, you can merge the finetuned model with the original model with the following code:

```sql
-- 合并lora model + base model

!byzerllm setup single;

run command as LLM.`` where 
action="convert"
and pretrainedModelType="deepspeed/llama"
and model_dir="/home/byzerllm/models/sft/jobs/sft-william-20230912-21-50-10-2529bf9f-493e-40a3-b20f-0369bd01d75d/finetune_model/final/pretrained_model"
and checkpoint_dir="/home/byzerllm/models/sft/jobs/sft-william-20230912-21-50-10-2529bf9f-493e-40a3-b20f-0369bd01d75d/finetune_model/final"
and savePath="/home/byzerllm/models/sft/jobs/sft-william-20230912-21-50-10-2529bf9f-493e-40a3-b20f-0369bd01d75d/finetune_model/merge";

```







