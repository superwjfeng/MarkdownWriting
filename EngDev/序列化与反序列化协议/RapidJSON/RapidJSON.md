# Intro

[Tencent/rapidjson: A fast JSON parser/generator for C++ with both SAX/DOM style API (github.com)](https://github.com/Tencent/rapidjson?tab=readme-ov-file)、

[RapidJSON: 首页](https://rapidjson.org/zh-cn/)

Tencent/rapidjson - 单纯的 JSON 库，甚至没依赖 STL

### 安装

```cmd
$ git clone --depth 1 --progress https://github.com/Tencent/rapidjson.git
$ cd rapidjson
$ cmake -S . -B build
$ cd build
$ make install
```

### Use RapidJSON with CMake

```cmake
find_package(RapidJSON REQUIRED CONFIG)
include_directories(${RapidJSON_INCLUDE_DIRS})

target_link_libraries(my_exe
  ${RapidJSON_LIBS}
)
```

# DOM & SAX风格API

其实DOM和SAX两种解析风格都是用于XML文件的（*序列化与反序列化协议.md*），RapidJSON借用了这两种名字来描述它的两种JSON解析器风格 

## *DOM*

JSON被解析到一个内存中的DOM-like（Document Object Model）结构。这意味着整个JSON文档被读取并转换成一个树状结构，其中包含了多种类型的节点，如对象、数组、字符串、数字等

RapidJSON 可把 JSON 解析至一个 DOM 表示方式（`rapidjson::GenericDocument`），以方便操作。如有需要，可把 DOM 转换（stringify）回 JSON

DOM 风格 API（`rapidjson::GenericDocument`）实际上是由 SAX 风格 API（`rapidjson::GenericReader`）实现的。SAX 更快，但有时 DOM 更易用。用户可根据情况作出选择



###  Parse

### 原位解析

> *In situ* ... is a Latin phrase that translates literally to "on site" or "in position". It means "locally", "on site", "on the premises" or "in place" to describe an event where it takes place, and is used in many different contexts. ... (In computer science) An algorithm is said to be an in situ algorithm, or in-place algorithm, if the extra amount of memory required to execute the algorithm is O(1), that is, does not exceed a constant no matter how large the input. For example, heapsort is an in situ sorting algorithm.
>
> *In situ*……是一个拉丁文片语，字面上的意思是指「现场」、「在位置」。在许多不同语境中，它描述一个事件发生的位置，意指「本地」、「现场」、「在处所」、「就位」。 …… （在计算机科学中）一个算法若称为原位算法，或在位算法，是指执行该算法所需的额外内存空间是 O(1) 的，换句话说，无论输入大小都只需要常数空间。例如，堆排序是一个原位排序算法。

原位解析 Insitu parse：一种空间复杂度为***O(1)***的解析方式。正常解析方式需要将JSON字符串复制到其他缓冲区进行解析，这样将会消耗时间和空间复杂度。而原位解析则在JSON字符串所在的原空间进行操作，效率比普通解析高

由于原位解析修改了输入，其解析 API 需要 `char*` 而非 `const char*`

## *SAX*

RapidJSON还提供SAX-style（Simple API for XML）的解析器。与DOM解析不同，SAX解析器是基于事件的，并且它在解析JSON文档时不会在内存中构建完整的树状结构。这使得SAX解析器非常快速且内存消耗低，但是也更难使用，因为需要处理解析过程中发生的事件

### Reader

RapidJSON 提供一个事件循序访问的解析器 API（`rapidjson::GenericReader`）

```C++
// reader.h 
namespace rapidjson {
 
template <typename SourceEncoding, typename TargetEncoding, typename Allocator = MemoryPoolAllocator<> >
class GenericReader {
    // ...
};
 
typedef GenericReader<UTF8<>, UTF8<> > Reader;
 
} // namespace rapidjson
```

Reader类是SAX风格的解析器，用于解析JSON文本，并通过一系列事件回调通知用户解析的进度。用户可以实现自己的处理程序来处理这些事件

### Writer

RapidJSON 也提供一个生成器 API（`rapidjson::Writer`），可以处理相同的事件集合。Writer & Reader的设计不太一样，Write不是一个 `typedef`，而是一个模板类

```C++
namespace rapidjson {
 
template<typename OutputStream, typename SourceEncoding = UTF8<>, typename TargetEncoding = UTF8<>, typename Allocator = CrtAllocator<> >
class Writer {
public:
    Writer(OutputStream& os, Allocator* allocator = 0, size_t levelDepth = kDefaultLevelDepth)
// ...
};
 
} // namespace rapidjson-
```

Writer类用于将JSON内容输出到流中，例如标准输出、文件或字符串。它可以与SAX接口一起使用，以便在解析JSON时直接写入输出流，而无需先构建DOM树

### PrettyWriter

Writer所输出的是没有空格字符的最紧凑JSON，适合网络传输或储存，但不适合人类阅读。因此RapidJSON提供了一个 PrettyWriter，它在输出中加入缩进及换行

PrettyWriter的用法与Writer几乎一样，不同之处是PrettyWriter提供了一个 `SetIndent(Ch indentChar, unsigned indentCharCount)`。缺省的缩进是4个空格

# 流

`FileReadStream` 只会从文件读取一部分至缓冲区，然后让那部分被解析。若缓冲区的字符都被读完，它会再从文件读取下一部分

## *iostream包装类*

* `IStreamWrapper` 把任何继承自 `std::istream` 的类（如 `std::istringstream`、`std::stringstream`、`std::ifstream`、`std::fstream`）包装成 RapidJSON 的输入流
* `OStreamWrapper` 把任何继承自 `std::ostream` 的类（如 `std::ostringstream`、`std::stringstream`、`std::ofstream`、`std::fstream`）包装成 RapidJSON 的输出流

## *自定义流*

# 架构 & 实现

<img src="architecture.png">



## *Value & Document*

```C++
namespace rapidjson {
 
template <typename Encoding, typename Allocator = MemoryPoolAllocator<> >
class GenericValue {
    // ...
};
 
template <typename Encoding, typename Allocator = MemoryPoolAllocator<> >
class GenericDocument : public GenericValue<Encoding, Allocator> {
    // ...
};
 
typedef GenericValue<UTF8<> > Value;
typedef GenericDocument<UTF8<> > Document;
 
} // namespace rapidjson
```

* Value是RapidJSON中最重要的类之一，代表JSON值的所有可能类型，包括`null`、布尔值、数字（整数和浮点数）、字符串、数组和对象。Value对象可以轻松地从一个类型转换为另一个类型，并且可以容纳复杂的嵌套结构
* Document类继承自Value类，并代表整个JSON文档。它通常作为DOM解析的起点，加载和存储整个JSON DOM树

### Move Semantics





​	

rapidjson为了最大化性能，大量使用了浅拷贝，使用之前一定要了解清楚。
如果采用了浅拷贝，特别要注意局部对象的使用，以防止对象已被析构了，却还在被使用



## *查询Value*

map风格的获取KV pair的value，不过此时仍然是RapidJSON的Value数据类型，还得通过 `GetXXX()` 来转换成某种具体的C++类型

通过 `IsXXX()` 来验证Value是否是某种类型

## *创建/修改值*

当使用默认构造函数创建一个 Value 或 Document，它的类型便会是 `Null`。要改变其类型，需调用 `SetXXX()` 或赋值操作

```C++
Document d; // Null
d.SetObject();
 
Value v;    // Null
v.SetInt(10);
v = 10;     // 简写，和上面的相同
```







