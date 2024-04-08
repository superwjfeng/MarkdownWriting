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