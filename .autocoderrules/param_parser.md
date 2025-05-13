

---
description: 参数解析与类型转换
globs: "src/byzerllm/__init__.py"
alwaysApply: false
---

# 参数解析器

## 简要说明
提供带前缀的参数解析功能，支持自动类型转换（int/float/bool/list/dict）。适用于从配置字典中提取和转换特定前缀的参数。

## 典型用法
```python
from byzerllm import parse_params

params = {
    "sft.float.learning_rate": "0.001",
    "sft.int.epochs": "10",
    "sft.bool.shuffle": "true",
    "sft.list.layers": "[1,2,3]",
    "sft.dict.optimizer": '{"name":"adam","lr":0.01}'
}

# 解析带sft前缀的参数
parsed = parse_params(params, prefix="sft")

# 结果会自动转换类型:
# {
#   "learning_rate": 0.001,  # float
#   "epochs": 10,            # int  
#   "shuffle": True,         # bool
#   "layers": [1, 2, 3],     # list
#   "optimizer": {"name": "adam", "lr": 0.01}  # dict
# }
```

## 依赖说明
- json: 用于解析list/dict类型的参数
- 支持的类型标记:
  - float: 转换为浮点数
  - int: 转换为整数
  - bool: 转换为布尔值 ("true"/"false")
  - list: JSON解析为列表
  - dict: JSON解析为字典
  - str: 保持字符串不变

## 学习来源
从src/byzerllm/__init__.py中的parse_params函数提取

