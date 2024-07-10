[Tencent/rapidjson: A fast JSON parser/generator for C++ with both SAX/DOM style API (github.com)](https://github.com/Tencent/rapidjson?tab=readme-ov-file)、

[RapidJSON: 首页](https://rapidjson.org/zh-cn/)



## *DOM & SAX*

### DOM

### SAX



### RapidJSON的两种API风格

其实DOM和SAX两种解析风格都是用于XML文件的，RapidJSON借用了这两种名字来描述它的两种JSON解析器风格

* DOM, Document Object Model

  JSON被解析到一个内存中的DOM-like结构。这意味着整个JSON文档被读取并转换成一个树状结构，其中包含了多种类型的节点，如对象、数组、字符串、数字等

* SAX, Simple API for XML

  RapidJSON还提供SAX-style的解析器。与DOM解析不同，SAX解析器是基于事件的，并且它在解析JSON文档时不会在内存中构建完整的树状结构。这使得SAX解析器非常快速且内存消耗低，但是也更难使用，因为你需要处理解析过程中发生的事件



Value是RapidJSON中最重要的类之一，代表JSON值的所有可能类型，包括`null`、布尔值、数字（整数和浮点数）、字符串、数组和对象。Value对象可以轻松地从一个类型转换为另一个类型，并且可以容纳复杂的嵌套结构



Document类继承自Value类，并代表整个JSON文档。它通常作为DOM解析的起点，加载和存储整个JSON DOM树



Writer类用于将JSON内容输出到流中，例如标准输出、文件或字符串。它可以与SAX接口一起使用，以便在解析JSON时直接写入输出流，而无需先构建DOM树

Reader类是SAX风格的解析器，用于解析JSON文本，并通过一系列事件回调通知用户解析的进度。用户可以实现自己的处理程序来处理这些事件
