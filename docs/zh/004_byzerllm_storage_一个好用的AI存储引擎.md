# Byzer Storage: 最易用的AI存储引擎

Byzer Storage是一个为AI应用设计的高性能存储引擎,它提供了简单易用的API,支持向量搜索、全文检索以及结构化查询。本文将详细介绍Byzer Storage的使用方法和主要特性。

## 0. 安装和启动

```bash
pip install byzerllm
byzerllm storage start
```

That's it! Byzer Storage已经安装并启动成功,现在我们可以开始使用它了。默认会启动一个 byzerai_store 的集群。

注意，如果你是这样启动的：

```bash
byzerllm storage start --enable_emb
```

那么会自动启动一个 emb 模型，名字就叫 emb， ByzerStorage 会自动使用该模型，无需做任何其他配置。

## 1. 初始化

创建一个 ByzerStorage 对象，链接 byzerai_store 集群，并且指定数据库和表名（可以不存在）。

```python
from byzerllm.apps.byzer_storage.simple_api import ByzerStorage, DataType, FieldOption, SortOption

storage = ByzerStorage("byzerai_store", "my_database1", "my_table4s")
```


## 2. 创建库表（可选）

Byzer Storage使用Schema来定义数据结构。我们可以使用SchemaBuilder来构建Schema:

```python
_ = (
    storage.schema_builder()
    .add_field("_id", DataType.STRING)
    .add_field("name", DataType.STRING)
    .add_field("content", DataType.STRING, [FieldOption.ANALYZE])
    .add_field("raw_content", DataType.STRING, [FieldOption.NO_INDEX])    
    .add_array_field("summary", DataType.FLOAT)    
    .add_field("created_time", DataType.LONG, [FieldOption.SORT])    
    .execute()
)
```

这个Schema定义了以下字段:
- `_id`: 字符串类型的主键
- `name`: 字符串类型,可用于过滤条件
- `content`: 字符串类型,带有ANALYZE选项,用于全文搜索
- `raw_content`: 字符串类型,带有NO_INDEX选项,不会被索引
- `summary`: 浮点数组类型,用于存储向量
- `created_time`: 长整型,带有SORT选项,可用于排序

需要注意的是：

1. 如果一个字段带有ANALYZE选项,则该字段会被分词,并且可以用于全文搜索，但是就无法返回原始的文本了。所以你需要添加一个新字段专门用来存储原文，比如在我们这个例子里，我们新增了 raw_content 字段，并且显示的指定了 NO_INDEX 选项，这样就不会被索引，也就不会被分词，可以后续被检索后用来获取原始文本。
2. 对于需要作为向量检索和存储的字段，需要指定为数组类型，比如我们这个例子里的 summary 字段。
3. 如果你需要拿到向量字段的原始文本，那么你也需要添加一个新字段专门用来存储原文，就像我们这个例子里的 raw_content 字段一样。


## 3. 写入数据

准备数据并使用WriteBuilder写入Storage:

```python
data = [
    {"_id": "1", "name": "Hello", "content": "Hello, world!", "raw_content": "Hello, world!", "summary": "hello world", "created_time": 1612137600},
    {"_id": "2", "name": "Byzer", "content": "Byzer, world!", "raw_content": "Byzer, world!", "summary": "byzer", "created_time": 1612137601},
    {"_id": "3", "name": "AI", "content": "AI, world!", "raw_content": "AI, world!", "summary": "AI", "created_time": 16121376002},
    {"_id": "4", "name": "ByzerAI", "content": "ByzerAI, world!", "raw_content": "ByzerAI, world!", "summary": "ByzerAi", "created_time": 16121376003},
]

storage.write_builder().add_items(data, vector_fields=["summary"], search_fields=["content"]).execute()
storage.commit()
```

这里我们使用`add_items`方法批量添加数据,并指定了`summary`为向量字段,`content`为搜索字段。最后调用`commit()`来确保数据被持久化。


从上面我们要写入到 byzer storage 的数据我们可以看到如下几个特点：

1. 需要包含我们之前定义的 Schema中罗列的所有的字段，同时需要指定哪些是向量字段，哪些是检索字段。
2. 向量和检索不能是同一个字段。
3. 对于向量，检索字段，我们给到 write_builder 的都是文本，ByzerStorage 会根据 Schema 的定义自动将其转换为向量，检索字段需要的格式。

## 4. 查询数据

Byzer Storage支持多种查询方式,包括向量搜索、全文检索、过滤和排序。

### 4.1 向量搜索 + 全文检索

```python
query = storage.query_builder()
query.set_vector_query("ByzerAI", fields=["summary"])
results = query.set_search_query("Hello", fields=["content"]).execute()
print(results)
```

这个查询结合了向量搜索和全文检索,它会在`summary`字段中搜索与"ByzerAI"相似的向量,同时在`content`字段中搜索包含"Hello"的文档。

### 4.2 过滤 + 向量搜索 + 全文检索

```python
query = storage.query_builder()
query.and_filter().add_condition("name", "AI").build()
query.set_vector_query("ByzerAI", fields="summary")
results = query.set_search_query("Hello", fields=["content"]).execute()
print(results)
```

这个查询首先过滤`name`字段等于"AI"的文档,然后在结果中进行向量搜索和全文检索。

### 4.3 过滤 + 排序

```python
query = storage.query_builder()
query.and_filter().add_condition("name", "AI").build().sort("created_time", SortOption.DESC)
results = query.execute()
print(results)
```

这个查询过滤`name`字段等于"AI"的文档,然后按`created_time`字段降序排序。

## 5. 删除数据

### 5.1 根据ID删除

```python
storage.delete_by_ids(["3"])

query = storage.query_builder()
query.and_filter().add_condition("name", "AI").build()
results = query.execute()
print(results)
```

这里我们删除了ID为"3"的文档,然后查询验证删除结果。

### 5.2 删除整个表

```python
storage.drop()
```

这个操作会删除整个表及其所有数据,请谨慎使用。

## 结论

Byzer Storage提供了一套简洁而强大的API,能够轻松实现向量搜索、全文检索、结构化查询等功能。它的设计非常适合AI应用场景,可以有效地存储和检索各种类型的数据。通过本文的介绍,相信读者已经对Byzer Storage有了基本的了解,并能够开始在自己的项目中使用这个强大的存储引擎。
