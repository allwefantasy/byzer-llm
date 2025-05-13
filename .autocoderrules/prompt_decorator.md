

---
description: 高级Prompt装饰器实现
globs: "src/byzerllm/__init__.py"
alwaysApply: false
---

# Prompt装饰器模式

## 简要说明
提供声明式Prompt管理，支持多轮对话、响应标记、类型转换和元数据处理。通过装饰器简化LLM交互，支持jinja2模板渲染和结果验证。

## 典型用法
```python
from byzerllm import prompt, ByzerLLM
from pydantic import BaseModel

# 定义返回类型
class UserInfo(BaseModel):
    name: str
    age: int

# 使用prompt装饰器
@prompt(llm="qwen", render="jinja2", check_result=True)
def extract_user_info(text: str) -> UserInfo:
    """从文本中提取用户信息
    Args:
        text: 包含用户信息的文本
    Returns:
        UserInfo对象
    """
    return {"text": text}

# 初始化LLM
llm = ByzerLLM()
llm.setup_default_model_name("qwen")

# 调用并获取结构化结果
result = extract_user_info.with_return_type(UserInfo) \
                         .with_response_markers(["<RESPONSE>", "</RESPONSE>"]) \
                         .run("John is 30 years old")
print(result.name)  # 输出: John
```

## 依赖说明
- byzerllm: 核心LLM库
- pydantic: 用于返回类型验证
- jinja2: 模板渲染 (可选)
- 功能特性:
  - 多轮对话支持
  - 自动响应标记处理
  - 类型安全返回
  - 元数据跟踪

## 学习来源
从src/byzerllm/__init__.py中的prompt类及相关实现提取

