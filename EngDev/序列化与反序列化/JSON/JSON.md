## *JSON语法*

JSON, JavaScript Object Notation 是一种轻量级的数据交换格式，易于人阅读和编写，同时也易于机器解析和生成。JSON 基于 JavaScript 语言标准（ECMA-262 第 3 版）的一个子集，但它是完全独立于 JavaScript 语言本身的，很多编程环境都支持 JSON

官方网站是：[JSON](https://www.json.org/json-en.html)

一个简单的例子是

```json
{
  "name": "John",
  "age": 30,
  "isStudent": false
}
```

**JSON 不支持注释**。尽管某些 JSON 解析器可能会容忍注释，但它们并不属于 JSON 标准的一部分

### JSON的数据类型

数据由逗号分隔：多个名称/值对用逗号分隔。在数组或对象中，每个元素或成员之间都使用逗号来分隔

* 基本类型
    * 字符串 string 必须用双引号：在 JSON 中，**字符串必须被双引号包围**。单引号不是有效的 JSON 语法
    * 数字 number，其值为int、double
    * 布尔值 bool（true or false）
    * `null`
* 复合类型
    * list类型（也称array类型或者数组）：用方括号 `[]` 包围array，表示array的开始和结束。array可以包含多个值，这些值可以是不同类型的，比如字符串、数字、对象或者其他array
    * dict类型（也称为对象 object）：大括号 `{}` 包围对象，表示对象开始和结束。在大括号内部，可以包含多个名称/值对，其中键（名称）是字符串。dict是JSON中最常用的表现形式

### 示例

下面是一个包含多种数据类型的 JSON 对象示例：

```json
{
  "firstName": "John",
  "lastName": "Doe",
  "age": 21,
  "isStudent": true,
  "courses": ["Math", "Science", "Literature"],
  "address": {
    "street": "123 Main St",
    "city": "Anytown",
    "zipCode": "12345"
  },
  "nullValue": null
}
```

## *Jsoncpp*

我们以Jsoncpp为例来说明一个JSON库一般都需要实现哪些核心数据结构和接口

[open-source-parsers/jsoncpp: A C++ library for interacting with JSON. (github.com)](https://github.com/open-source-parsers/jsoncpp)

```
.
├── allocator.h
├── assertions.h
├── config.h
├── forwards.h
├── json_features.h
├── json.h
├── reader.h
├── value.h
├── version.h
└── writer.h
```

上面是Jsoncpp的实现文件，基本上所有的JSON的C++库的核心实现就是Object、Reader和Writer



* `is*()`：数据类型的判断
* `as*()`：数据类型的显性转换
* `has()`：JSON对象中有没有
* `remove()`：删除



### 需要支持的功能

* 序列化 Serializing：将编程语言中的数据结构转换为JSON格式的字符串，以便进行存储或网络传输
* Parsing 解析/Deserializing 反序列化：将JSON格式的字符串解析成1999999程语言中的相应数据结构（如JavaScript中的对象或数组）
* 读取 Reader：从文件系统、Web API响应或其他来源读取JSON数据
* 写入 Writer：将JSON数据写入文件系统或通过网络发送
* 验证 Validation：验证JSON数据的结构是否符合特定的模式或规范
* 查询 Query：查询JSON文档以检索特定的元素或值
* 转换 Transformation：对JSON数据进行转换，如过滤、映射、排序等操作

 ### Value类

JSON首先要有一种能用JSON规定的数据格式来表示C++中的对应（可能有多种的）数据类型的能力，这种能力一般都是通过Value类来实现的

### Reader类

`Json:Reader` 类顾名思义主要用于读取：它可以将字符串转换成 `Json:Value` 对象，或者反过来从一个json文件中读取内容然后转成 `Json::Value` 对象

### Writer类



## *Rapidjson*

Tencent/rapidjson - 单纯的 JSON 库，甚至没依赖 STL

## *YYJson*

[ibireme/yyjson: The fastest JSON library in C (github.com)](https://github.com/ibireme/yyjson)

## *Python的JSON包*

Python 的 `json` 模块提供了一种简单的方式来编码和解码JSON数据

在 Python 脚本中，首先需要导入 `json` 模块：

```Python
import json
```

### Python数据类型 & JSON数据类型的对应关系

Python `<-->` JSON

- dict `<-->` object
- list, tuple `<-->` array
- str `<-->` string
- int, float, int- & float-derived Enums `<-->` number
- True `<-->` true
- False `<-->` false
- None `<-->` null

### 将 Python 对象编码成 JSON 字符串（序列化）

- `json.dumps()`: 将 Python 对象转换成 JSON 格式的字符串

  ```python
  data = {
      'name': 'John Doe',
      'age': 30,
      'is_employee': True,
      'titles': ['Developer', 'Engineer']
  }
  
  json_string = json.dumps(data)
  print(json_string)  # 输出 JSON 格式的字符串
  ```

- `json.dump()`: 将 Python 对象转换成 JSON 格式的字符串，并将其写入到一个文件中

  ```python
  with open('data.json', 'w') as outfile:
      json.dump(data, outfile)
  # 这会创建 data.json 文件，并写入 JSON 数据
  ```

### 将 JSON 字符串解码成 Python 对象（反序列化）

- `json.loads()`: 将 JSON 格式的字符串解码成 Python 对象

  ```python
  json_string = '{"name": "John Doe", "age": 30, "is_employee": true, "titles": ["Developer", "Engineer"]}'
  
  data = json.loads(json_string)
  print(data)  # 输出解码后的 Python 字典
  ```

- `json.load()`: 读取一个文件，并将其中的 JSON 字符串解码成 Python 对象

  ```python
  with open('data.json', 'r') as infile:
      data = json.load(infile)
  # 从 data.json 读取内容，并转换成 Python 对象
  ```

### 高级选项

`json.dumps()` 和 `json.dump()` 方法接受多个可选参数，以定制编码过程：

- `indent`: 指定缩进级别，用于美化输出。例如，`indent=4` 会用四个空格缩进。
- `separators`: 指定分隔符，默认是 `(', ', ': ')`。如果你想让输出更紧凑，可以使用 `(',', ':')`。
- `sort_keys`: 当设置为 `True` 时，字典的键将被排序。

```python
json_string = json.dumps(data, indent=4, sort_keys=True)
print(json_string)
```

同样，`json.loads()` 和 `json.load()` 也有参数来处理特定情况，比如解析不符合 JSON 规范的数据

### Pickle 包

python的pickle和json一样，两者都有dumps、dump、loads、load四个API

* json包用于字符串和python数据类型间进行转换
* pickle用于python特有的类型和python的数据类型间进行转换

json是可以在不同语言之间交换数据的,而pickle只在python之间下使用

json只能序列化最基本的数据类型，而pickle可以序列化所有的数据类型，包括类、函数都可以序列化

https://blog.csdn.net/ITBigGod/article/details/86477083

>The [`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle) module implements binary protocols for serializing and de-serializing a Python object structure. *“Pickling”* is the process whereby a Python object hierarchy is converted into a byte stream, and *“unpickling”* is the inverse operation, whereby a byte stream (from a [binary file](https://docs.python.org/3/glossary.html#term-binary-file) or [bytes-like object](https://docs.python.org/3/glossary.html#term-bytes-like-object)) is converted back into an object hierarchy. Pickling (and unpickling) is alternatively known as “serialization”, “marshalling,” [1](https://docs.python.org/3/library/pickle.html#id7) or “flattening”; however, to avoid confusion, the terms used here are “pickling” and “unpickling”.

用 `pickle.load()` 进行序列化 serializing or pickling 和 `pickle.dump()` 进行反序列化 de-serializing or unpickling

下面的例子是i2dl中的MemoryImageFolderDataset类，用于将不大的Dataset放到内存中，来加快IO速度

```python
with open(os.path.join(self.root_path, 'cifar10.pckl'), 'rb') as f:
    save_dict = pickle.load(f)
```
