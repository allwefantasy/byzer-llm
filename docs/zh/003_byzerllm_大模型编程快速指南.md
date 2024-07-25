##  byzerllm 大模型编程快速指南

本文示例 [Notebook](../../notebooks/003_byzerllm_大模型编程快速指南.ipynb)

## 安装

```bash
pip install -U byzerllm
ray start --head
```
## 启动一个模型代理

byzerllm 支持私有化模型或者SaaS模型的部署。

这里以 deepseek 官方API 为例：

```bash
easy-byzerllm deploy deepseek-chat --token xxxxx --alias deepseek_chat
```

或者跬基流动API:

```bash
easy-byzerllm deploy alibaba/Qwen1.5-110B-Chat --token xxxxx --alias qwen110b_chat
```

将上面的 API KEY 替换成你们自己的。

如果你想部署私有化模型或者对接 Ollama 等更多需求，参考 [002_使用byzerllm进行模型部署.md](./002_使用byzerllm进行模型部署.md) 或者 [README.md](../../README.md)。

之后，你就可以在代码里使用  deepseek_chat 或者 qwen110b_chat  访问模型了。

## hello world

来和我们的大模型打个招呼:

```python
import byzerllm

llm = byzerllm.ByzerLLM.from_default_model(model="deepseek_chat")

@byzerllm.prompt(llm=llm)
def hello(q:str) ->str:
    '''
    你好, {{ q }}
    '''

s = hello("你是谁")    
print(s)

## 输出:
## '你好！我是一个人工智能助手，专门设计来回答问题、提供信息和帮助解决问题。如果你有任何疑问或需要帮助，请随时告诉我。'
```

恭喜，你和大模型成功打了招呼！

可以看到，我们通过 `@byzerllm.prompt` 装饰器，将一个方法转换成了一个大模型的调用，然后这个方法的主题是一段文本，文本中
使用了 jinja2 模板语法，来获得方法的参数。当正常调用该方法时，实际上就发起了和大模型的交互，并且返回了大模型的结果。

在 byzerllm 中，我们把这种方法称之为 prompt 函数。

## 查看发送给大模型的prompt

很多情况你可能需要调试，查看自己的 prompt 渲染后到底是什么样子的，这个时候你可以通过如下方式
获取渲染后的prompt:

```python
hello.prompt("你是谁")
## '你好, 你是谁'
```            

## 动态换一个模型

前面的 hello 方法在初始化的时候，我们使用了默认的模型 deepseek_chat，如果我们想换一个模型，可以这样做：

```python
hello.with_llm(llm).run("你是谁")
## '你好！我是一个人工智能助手，专门设计来回答问题、提供信息和帮助解决问题。如果你有任何疑问或需要帮助，请随时告诉我。'
```

通过 with_llm 你可以设置一个新的 llm 对象，然后调用 run 方法，就可以使用新的模型了。

## 超长文本生成

我们知道，大模型一次生成的长度其实是有限的，如果你想生成超长文本，你可能需手动的不断获得
生成结果，然后把他转化为输入，然后再次生成，这样的方式是比较麻烦的。

byzerllm 提供了更加易用的 API :

```python
import byzerllm
from byzerllm import ByzerLLM

llm = ByzerLLM.from_default_model("deepseek_chat")

@byzerllm.prompt()
def tell_story() -> str:
    """
    讲一个100字的故事。    
    """


s = (
    tell_story.with_llm(llm)
    .with_response_markers()
    .options({"llm_config": {"max_length": 10}})
    .run()
)
print(s)

## 从前，森林里住着一只聪明的小狐狸。一天，它发现了一块闪闪发光的宝石。小狐狸决定用这块宝石帮助森林里的动物们。它用宝石的光芒指引迷路的小鹿找到了回家的路，用宝石的温暖治愈了受伤的小鸟。从此，小狐狸成了森林里的英雄，动物们都感激它的善良和智慧。
```

实际核心部分就是这一行：

```python
tell_story.with_llm(llm)
    .with_response_markers()    
    .run()
```

我们只需要调用 `with_response_markers` 方法，系统就会自动的帮我们生成超长文本。
在上面的案例中，我们通过

```python
.options({"llm_config": {"max_length": 10}})
```

认为的限制大模型一次交互最多只能输出10个字符，但是系统依然自动完成了远超过10个字符的文本生成。

## 对象输出

前面我们的例子都是返回字符串，但是我们也可以返回对象，这样我们就可以更加灵活的处理返回结果。

```python
import pydantic 

class Story(pydantic.BaseModel):
    '''
    故事
    '''

    title: str = pydantic.Field(description="故事的标题")
    body: str = pydantic.Field(description="故事主体")

@byzerllm.prompt()
def tell_story()->Story:
    '''
    讲一个100字的故事。    
    '''

s = tell_story.with_llm(llm).run()
print(isinstance(s, Story))
print(s.title)

## True
## 勇敢的小鸟
```

可以看到，我们很轻松的将输出转化为格式化输出。

## 自定义字段抽取

前面的结构化输出，其实会消耗更多token,还有一种更加精准的结构化输出方式。
比如让大模型生成一个正则表达式，但实际上大模型很难准确只输出一个正则表达式，这个时候我们可以通过自定义抽取函数来获取我们想要的结果。


```python
from loguru import logger
import re

@byzerllm.prompt()
def generate_regex_pattern(desc: str) -> str:
    """
    根据下面的描述生成一个正则表达式，要符合python re.compile 库的要求。

    {{ desc }}

    最后生成的正则表达式要在<REGEX></REGEX>标签对里。
    """    

def extract_regex_pattern(regex_block: str) -> str:    
    pattern = re.search(r"<REGEX>(.*)</REGEX>", regex_block, re.DOTALL)
    if pattern is None:
        logger.warning("No regex pattern found in the generated block:\n {regex_block}")
        raise None
    return pattern.group(1)

pattern = "匹配一个邮箱地址"
v = generate_regex_pattern.with_llm(llm).with_extractor(extract_regex_pattern).run(desc=pattern)
print(v)
## ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$
```

在上面的例子里，我们根据一句话生成一个正则表达式。我们通过 `with_extractor` 方法，传入了一个自定义的抽取函数，这个函数会在大模型生成结果后，对结果进行处理，然后返回我们想要的结果。

我们在 prompt 明确说了，生成的结果要放到 `<REGEX></REGEX>` 标签对里，然后我们通过 extract_regex_pattern 函数，从结果中提取出了我们想要的正则表达式。

## 在实例方法中使用大模型

```python
import byzerllm
data = {
    'name': 'Jane Doe',
    'task_count': 3,
    'tasks': [
        {'name': 'Submit report', 'due_date': '2024-03-10'},
        {'name': 'Finish project', 'due_date': '2024-03-15'},
        {'name': 'Reply to emails', 'due_date': '2024-03-08'}
    ]
}


class RAG():
    def __init__(self):        
        self.llm = byzerllm.ByzerLLM()
        self.llm.setup_template(model="deepseek_chat",template="auto")
        self.llm.setup_default_model_name("deepseek_chat")        
    
    @byzerllm.prompt(lambda self:self.llm)
    def generate_answer(self,name,task_count,tasks)->str:
        '''
        Hello {{ name }},

        This is a reminder that you have {{ task_count }} pending tasks:
        {% for task in tasks %}
        - Task: {{ task.name }} | Due: {{ task.due_date }}
        {% endfor %}

        Best regards,
        Your Reminder System
        '''        

t = RAG()

response = t.generate_answer(**data)
print(response)

## 输出:
## Hello Jane Doe,
##I hope this message finds you well. I wanted to remind you of your 3 pending tasks to ensure you stay on track:
## 1. **Submit report** - This task is due on **2024-03-10**. Please ensure that you allocat
```

这里我们给了个比较复杂的例子，但我们可以看到，给一个实例prompt方法和普通prompt 方法差异不大。
唯一的区别是如果你希望在定义的时候就指定大模型，使用一个lambda函数返回实例的 llm 对象即可。

```python
@byzerllm.prompt(lambda self:self.llm)
```

你也可以不返回，在调用的时候通过 `with_llm` 方法指定 llm 对象。

此外，这个例子也展示了如何通过jinja2模板语法，来处理复杂的结构化数据。

## 通过 Python 代码处理复杂入参

上面的一个例子中，我们通过 jinja2 模板语法，来处理复杂的结构化数据，但是有时候我们可能需要更加复杂的处理，这个时候我们可以通过 Python 代码来处理。

```python
import byzerllm

data = {
    'name': 'Jane Doe',
    'task_count': 3,
    'tasks': [
        {'name': 'Submit report', 'due_date': '2024-03-10'},
        {'name': 'Finish project', 'due_date': '2024-03-15'},
        {'name': 'Reply to emails', 'due_date': '2024-03-08'}
    ]
}


class RAG():
    def __init__(self):        
        self.llm = byzerllm.ByzerLLM.from_default_model(model="deepseek_chat")
    
    @byzerllm.prompt()
    def generate_answer(self,name,task_count,tasks)->str:
        '''
        Hello {{ name }},

        This is a reminder that you have {{ task_count }} pending tasks:
            
        {{ tasks }}

        Best regards,
        Your Reminder System
        '''
        
        tasks_str = "\n".join([f"- Task: {task['name']} | Due: { task['due_date'] }" for task in tasks])
        return {"tasks": tasks_str}

t = RAG()

response = t.generate_answer.with_llm(t.llm).run(**data)
print(response)

## Just a gentle nudge to keep you on track with your pending tasks. Here's a quick recap:....
```

在这个例子里，我们直接把 tasks 在方法体里进行处理，然后作为一个字符串返回，最够构建一个字典，字典的key为 tasks,然后
你就可以在 docstring 里使用 `{{ tasks }}` 来引用这个字符串。

这样对于很复杂的入参，就不用谢繁琐的 jinja2 模板语法了。

## 如何自动实现一个方法

比如我定义一个签名，但是我不想自己实现里面的逻辑，让大模型来实现。这个在 byzerllm 中叫 function impl。我们来看看怎么
实现:

```python
import pydantic
class Time(pydantic.BaseModel):
    time: str = pydantic.Field(...,description="时间，时间格式为 yyyy-MM-dd")


@llm.impl()
def calculate_current_time()->Time:
    '''
    计算当前时间
    '''
    pass 


calculate_current_time()
#output: Time(time='2024-06-14')
```

在这个例子里，我们定义了一个 calculate_current_time 方法，但是我们没有实现里面的逻辑，我们通过 `@llm.impl()` 装饰器，让大模型来实现这个方法。
为了避免每次都要“生成”这个方法，导致无法适用，我们提供了缓存，用户可以按如下方式打印速度：

```python
start = time.monotonic()
calculate_current_time()
print(f"first time cost: {time.monotonic()-start}")

start = time.monotonic()
calculate_current_time()
print(f"second time cost: {time.monotonic()-start}")

# output:
# first time cost: 6.067266260739416
# second time cost: 4.347506910562515e-05
```
可以看到，第一次执行花费了6s,第二次几乎是瞬间完成的，这是因为第一次执行的时候，我们实际上是在生成这个方法，第二次执行的时候，我们是执行已经生成好的代码，所以速度会非常快。你可以显示的调用 `llm.clear_impl_cache()` 清理掉函数缓存。

## Stream 模式

前面的例子都是一次性生成结果，但是有时候我们可能需要一个流式的输出，这个时候我们可能需要用底层一点的API来完成了：

```python
import byzerllm

llm = byzerllm.ByzerLLM.from_default_model(model="deepseek_chat")

v = llm.stream_chat_oai(model="deepseek_chat",conversations=[{
    "role":"user",
    "content":"你好，你是谁",
}],delta_mode=True)

for t in v:
    print(t,flush=True)  

# 你好
# ！
# 我
# 是一个
# 人工智能
# 助手
# ，
# 旨在
# 提供
# 信息
# 、
# 解答
# 问题....
```

如果你不想要流式输出，但是想用底层一点的API，你可以使用 `llm.chat_oai` 方法：

```python
import byzerllm

llm = byzerllm.ByzerLLM.from_default_model(model="deepseek_chat")

v = llm.chat_oai(model="deepseek_chat",conversations=[{
    "role":"user",
    "content":"你好，你是谁",
}])

print(v[0].output)
## 你好！我是一个人工智能助手，旨在提供信息、解答问题和帮助用户解决问题。如果你有任何问题或需要帮助，请随时告诉我。
```

## Function Calling 

byzerllm 可以不依赖模型自身就能提供 function calling 支持，我们来看个例子：


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

我们定义了两个方法，一个是计算日期范围，一个是计算当前时间。

现在我么可以来测试下，系统如何根据自然语言决定调用哪个方法：

```python
t = llm.chat_oai([{
    "content":'''计算当前时间''',
    "role":"user"    
}],tools=[compute_date_range,compute_now],execute_tool=True)

t[0].values

## output: ['2024-06-14 15:18:02']
```

我们可以看到，他正确的选择了 compute_now 方法。

接着我们再试一个：

```python
t = llm.chat_oai([{
    "content":'''最近三个月趋势''',
    "role":"user"    
}],tools=[compute_date_range,compute_now],execute_tool=True)

t[0].values

## output: [['2024-03-14 15:19:13', '2024-06-14 15:19:13']]
```

模型正确的选择了 compute_date_range 方法。

## 多模态

byerllm 也能很好的支持多模态的交互，而且统一了多模态大模型的接口，比如你可以用一样的方式使用 openai 或者 claude 的图片转文字能力， 或者一致的方式使用火山，azuer, openai的语音合成接口。

### image2text





## 注意事项

1. prompt函数方法体返回只能是dict，实际的返回类型和方法签名可以不一样，但是方法体返回只能是dict。
2. 大部分情况prompt函数体为空，如果一定要有方法体，可以返回一个空字典。
3. 调用prompt方法的时候，如果在@byzerllm.prompt()里没有指定llm对象，那么需要在调用的时候通过with_llm方法指定llm对象。






