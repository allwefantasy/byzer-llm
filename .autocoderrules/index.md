# Rules索引

本文档记录项目中所有可用的代码规则(rules)及其用途。

## .autocoderrules/llm_prompt_decorator.md
提供一个 Python 装饰器 (`@prompt`)，用于将带有类型注解和 Jinja2 模板化文档字符串的函数转换为结构化的 LLM API 调用。简化了提示词构建、API 请求发送（同步和流式）以及结果解析（字符串、Pydantic 模型或生成器）。