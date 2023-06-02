---
Title: Protobuf
Author: Weijian Feng 封伟健
Source: Official documentation for gRPC <https://grpc.io/docs/>, 官方文档中文翻译 <https://doc.oschina.net/grpc?t=56831>
---

C++用命名空间，Java用外部类进行统一爹管理

# Protobuf

## *intro*

Protocol Buffers (protobuf) 是一种由Google开发的二进制数据序列化格式。它旨在用于高效地序列化结构化数据，并支持跨不同平台、不同语言的数据交换

Protobuf 提供了一种定义数据结构的语言，称为 Protocol Buffers Language，通过该语言可以定义消息类型和相关字段。定义的消息类型被编译器转换成特定语言的类或结构体，以便在程序中使用

以下是 Protobuf 的一些主要特点

* 简洁性：Protobuf 的消息定义简洁明了，只需要定义字段的名称和类型即可。它支持基本数据类型（如整数、浮点数、布尔值等）以及复杂的数据结构（如嵌套消息、枚举等）
* 可扩展性：当需要向现有消息类型添加新字段时，可以保持向前和向后兼容性。旧版本的解析器可以解析新版本的消息，而新版本的解析器也可以解析旧版本的消息，只是忽略掉未知的字段
* 跨语言支持：Protobuf 支持多种编程语言，如C++、Java、Python、Go等。通过使用相应的编译器，可以将消息定义转换成目标语言的类或结构体，并在不同的语言之间进行数据交换
* 高效性：由于 Protobuf 使用二进制编码，因此相比于一些文本格式（如JSON、XML）具有更高的序列化和反序列化性能，以及更小的数据体积
* 可读性：虽然 Protobuf 的消息是以二进制格式存储的，但它同时支持可读的文本表示。可以使用 .proto 文件定义消息类型，并通过编译器生成用于序列化和反序列化的代码

在使用 Protobuf 进行数据交换时，需要先定义消息类型，并通过**编译器**生成对应的类或结构体。然后，通过序列化将消息对象转换为二进制数据，或通过反序列化将二进制数据转换为消息对象。这使得在不同的系统之间传输和存储结构化数据变得更加简单和高效

需要注意的是，由于 Protobuf 是一种二进制格式，因此对于人类可读性较差，相比于文本格式（如JSON）更适合机器间的数据交换

### 使用特点

**ProtoBuf需要依赖通过编译生成的头文件和源文件使用的**

* 编写 .proto 文件，目的是为了定义结构对象 message 及属性内容
* 使用 protoc 编译器编译 .proto 文件，生成一系列的接口代码，存放在新生成的头文件和源文件中
* 依赖生成的接口，将编译生成的头文件包含进我们的代码中，实现对 .proto 文件中定义的字段进行设置和获取，和对 message 对象进行序列化和反序列化

## *demo*

### 定义消息字段

<img src="ScalarValueType.png">

在 message 中可以定义其属性字段，字段定义格式为：`字段类型 字段名 = 字段唯一编号;`

* 字段名称命名规范：全小写字母，多个字母之间用 `_` 隔开
* 字段类型分为：标量数据类型 Scalar value type 和 特殊类型（枚举、map）
* 字段唯一编号：用来标识字段，一旦使用了就不能改变

变长编码：编译后可能4字节变1字节，体积变小了。负值的话会变成10字节

Sint32的效率更高一点



字段唯一编号的范围是 `1 ~ 536,870,911 (2^29 - 1)` ，其中 19000 ~ 19999 不可用。

19000 ~ 19999 不可用是因为：在 Protobuf 协议的实现中，对这些数进行了预留。如果非要在.proto 文件中使用这些预留标识号，例如将 name 字段的编号设置为19000，编译时就会报警

范围为 1 ~ 15 的字段编号需要一个字节进行编码， 16 ~ 2047 内的数字需要两个字节进行编码。编码后的字节不仅只包含了编号，还包含了字段类型。所以 1 ~ 15 要用来标记出现非常频繁的字段，要为将来有可能添加的、频繁出现的字段预留一些出来

### 编译proto文件

`-I` 指定搜索目录，不带的话就默认从当前目录搜索



把get给省略了，字段名函数就是get方法



Message的父类MessageLite类中定义了一系列的序列化与反序列化方法

* 编译成C++

  ```shell
  protoc --cpp_out=. contacts.proto
  ```

* 编译成Java

  ```shell
  protoc --java_out=. contacts.proto
  ```

  

```shell
g++ -o TestPb main.cc contacts.pb.cc -std=c++11 -lprotobuf
```

## *语法详解*

### 字段规则

* singular：消息中可以包含该字段零次或一次（不超过一次）。proto3 语法中，字段默认使用该规则
* repeated：消息中可以包含该字段任意多次（包括零次），其中重复值的顺序会被保留。可以理解为定义了一个数组



允许嵌套message

### demo 2.0

* 不再打印联系人的序列化结果，而是将通讯录序列化后并写入文件中
* 从文件中将通讯录解析出来，并进行打印
* 新增联系人属性，共包括：姓名、年龄、电话信息

### decode

```shell
❯ protoc --decode=contacts.Contacts contacts.proto < contacts.bin
Type not defined: contacts.Contacts
❯ protoc --decode=contacts2.Contacts contacts.proto < contacts.bin
contacts {
  name: "zhangsan"
  age: 20
  phone {
    number: "82"
  }
  phone {
    number: "138"
  }
}
```

### 枚举

反序列化时protobuf会对之前没有设置枚举值的字段默认设置为0

### any

可以将any类型理解为一种泛型，可以用来存储任意的数据类型

先定义好message，再用 `PackFrom()` 方法转换

### oneof

保证唯一性

不可以使用repeated

### 默认值

虽然标量没有has方法，但大多数情况下结合业务的语意信息是可以兼容其默认值的



### 更新消息

* 新增：不要和已有字段的名字和编号冲突即可
* 修改
* 删除：不可以直接删除，会造成数据损坏、错位等问题。被删除的字段要用 `reserved` 保留，之后不能被使用

不认识的字段不会被删除，而是会存储到未知字段

未知字段不能直接通过get获取

遍历UnkownFieldSet



## *通讯录v4-网络通讯录*

### 流程

1. 定义不同request和response的message
2. client序列化req
3. client调用req发送
4. server反序列化req
5. server业务处理
6. server序列化response
7. server调用response发送
8. client反序列化response
9. client业务处理
10. 回到第一步循环

## *序列化性能对比*

