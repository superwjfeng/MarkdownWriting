# LLVM IR/LLVM Assembly

[A Gentle Introduction to LLVM IR · mcyoung](https://mcyoung.xyz/2023/08/01/llvm-ir/)

[StormQ's Blog (csstormq.github.io)](https://csstormq.github.io/blog/LLVM 之 IR 篇（1）：零基础快速入门 LLVM IR)

LLVM 可以被认为是 Clang 的优化器和后端，这可以被认为是 “LLVM 语言” 或 “LLVM 汇编” 的编译器。Clang 和其他语言前端（如 Rust）本质上编译为 LLVM IR，然后 LLVM 将其编译为机器代码

LVM IR 是LLVM增强性、优化性和灵活性的核心所在。在编译过程中，源代码首先被转换成LLVM IR，然后经过各种优化和变换处理，最终生成目标平台的机器代码。LLVM的模块化和可扩展特性使得开发者可以很容易地实现新的优化、目标后端、新语言前端并通过共享的LLVM IR进行交互。这使得LLVM成为现代编译器构建的标准工具之一

## *LLVM IR 的格式*

LLVM IR 采用静态单赋值 SSA 的中间表示。它设计用于优化、分析和生成高度优化的机器码。LLVM IR有三种表现形式：内存中表示（In-memory）、字节码（Bitcode）文件和人类可读的文本表示。下面主要介绍其文本表示格式，因为这是开发者最常接触到的形式

### LLVM IR 的核心结构

* **模块（Module）**
  - 模块是LLVM IR的顶层容器，包含全局变量、函数和符号表等定义。
  - 通常是由一个源文件编译生成的
* **类型系统（Type System）**
  - 支持基本类型（如`i32`、`float`）、复合类型（如数组、结构体）、指针类型以及函数类型等。
* **函数 Function**：一个函数包含参数列表、返回类型和一系列基本块（Basic Blocks）
* **基本块 Basic Block**
  - 基本块是一系列顺序执行的LLVM指令，以终止指令（如`br`或`ret`）结尾
  - 每个基本块都有一个唯一标识符（Label）
* **指令（Instructions）**
  - LLVM指令以SSA形式组织，共享一个无限个寄存器的抽象
  - 常见指令包括`add`、`mul`、`load`、`store`、`phi`等













- `@` 全局标识
- `%` 局部标识
- `alloca` 开辟空间
- `align` 内存对齐
- `i32` 32个bit, 4个字节
- `store` 写入内存
- `load` 读取数据
- `call` 调用函数
- `ret` 返回



### 类型系统





## *IR的数据结构*

[看看 LLVM 的码（一）基础数据结构、IR (glass-panel.info)](https://blog.glass-panel.info/post/read-llvm-code-1/)

[LLVM笔记(16) - IR基础详解(一) underlying class - Five100Miles - 博客园 (cnblogs.com)](https://www.cnblogs.com/Five100Miles/p/14083814.html)

## *bitcode*

[LLVM Bitcode File Format — LLVM 19.0.0git documentation](https://llvm.org/docs/BitCodeFormat.html)

[blog/articles/llvm/2020_11_23_bc.md at main · zxh0/blog (github.com)](https://github.com/zxh0/blog/blob/main/articles/llvm/2020_11_23_bc.md)

bitcode 是 LLVM IR 的二进制形式

```cmd
$ clang -emit-llvm -c test.cc -o test.bc
$ file test.bc
test.bc: LLVM IR bitcode
```

### `-fembed-bitcode`

`-fembed-bitcode` 是 Clang 编译器的一个选项，它用于生成包含 LLVM bitcode 的二进制文件。当你在编译阶段使用这个选项时，编译器会在生成的目标文件（通常是 `.o` 文件）中嵌入一个 LLVM bitcode 的副本。这样，目标文件既包含了原生的机器码，也包含了可以进行进一步优化的 LLVM bitcode。

这个特性主要有两个应用场景：

1. **App Store 提交**：对于 iOS 和 macOS 应用开发者来说，当向 App Store 提交应用程序时，苹果公司要求所有的代码都包含 bitcode。这使得苹果可以重新优化应用程序而无需开发者提交新的二进制版本。例如，如果苹果发布了一个新的处理器或为现有的处理器引入了新的优化技术，他们可以自己对已经上传的 bitcode 进行重编译和优化，以便利用这些新特性
2. **后期优化和分析**：在其他环境，嵌入 bitcode 使得开发者能够在不访问源代码的情况下对程序进行后期的优化和分析。这可以用于产品支持、性能优化、安全分析等方面

还有几个相关的选项：

- `-fembed-bitcode-marker`：这个选项会在目标文件中插入一个占位符，代表 bitcode，但实际上并不包含真正的 bitcode 数据。这对确保工具链在其余部分支持 bitcode，但不需要完整的 bitcode 数据时非常有用。
- `-fembed-bitcode=all`：确保所有生成的文件（包括链接后的最终二进制文件）都包含 bitcode。

应当注意的是，嵌入 bitcode 会增加生成的二进制文件的大小，因为它既包含了可直接运行的机器码，也包含了用于可能的未来优化的 bitcode

# 代码优化 Pass

LLVM 中优化工作是通过一个个的 Pass（遍）来实现的，它支持三种类型的 Pass

1. 分析型 pass（Analysis Passes），只是做分析，产生一些分析结果用于后序操作
2. 做代码转换的 pass（Transform Passes），比如做公共子表达式删除
3. 工具型的 pass，比如对模块做正确性验证



# Clang Static Analyzer

[Clang Static Analyzer — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/ClangStaticAnalyzer.html)

Clang Static Analyzer，下面简称CSA，是LLVM提供的静态分析工具

[Clang Static Analyzer 介绍 | jywhy6's blog](https://blog.jywhy6.zone/2021/05/31/clang-static-analyzer-intro/)

CSA 是基于libclang实现的





整个 clang static analyzer 的入口是 AnalysisConsumer，接着会调 HandleTranslationUnit() 方法进行 AST 层级进行分析或者进行 path-sensitive 分析。默认会按照 inline 的 path-sensitive 分析，构建 CallGraph，从顶层 caller 按照调用的关系来分析，具体是使用的 WorkList 算法，从 EntryBlock 开始一步步的模拟，这个过程叫做 intra-procedural analysis（IPA）。这个模拟过程还需要对内存进行模拟，clang static analyzer 的内存模型是基于《A Memory Model for Static Analysis of C Programs》这篇论文而来，pdf地址：http://lcs.ios.ac.cn/~xuzb/canalyze/memmodel.pdf 在clang里的具体实现代码可以查看这两个文件 [MemRegion.h](https://code.woboq.org/llvm/clang/include/clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h.html)和 [RegionStore.cpp](https://code.woboq.org/llvm/clang/lib/StaticAnalyzer/Core/RegionStore.cpp.html) 。

## *内存模型*



## *Exploded Graph*

### CSA流程

<img src="CSA流程.drawio.png">

1. CSA以源代码为起点，将源代码转换为AST
2. 将AST转换为控制流图 CFG
3. 随着程序的模拟执行，Clang 的符号执行引擎会生成 Exploded Graph 扩展图，详细记录程序的执行位置和程序当前状态信息
4. 最后，在各个 Checker（CSA中可自定义的漏洞检查器）回调函数检测到漏洞产生时，将基于 Exploded Graph 中的数据生成带漏洞触发路径的漏洞报告

### Exploded Graph

[clang static analyzer源码分析（一）_clang源码分析-CSDN博客](https://blog.csdn.net/dashuniuniu/article/details/50773316)

[clang static analyzer源码分析（二）_clang源码分析-CSDN博客](https://blog.csdn.net/dashuniuniu/article/details/52434781)

### CSA的符号执行

## *Checker*

以alpha开头的checker是实验版本

- DeadStores: 检测未被使用的存储，即赋值给变量但随后未使用的值。
- MallocChecker: 检测与动态内存分配相关的问题，包括内存泄漏、双重释放等。
- NullDereference: 检测潜在的空指针解引用。
- DivideZero: 检测可能导致除零错误的情况。
- UninitializedObject: 检测可能未初始化的对象使用

## *自定义Checker*

[Checker Developer Manual (llvm.org)](https://clang-analyzer.llvm.org/checker_dev_manual.html)

# JIT Compiler

JIT Compiler（Just-In-Time Compiler）即时编译器，**它在运行时（即程序执行期间）将程序的源代码或字节码动态地编译成机器码，然后立即执行**。这与传统的AOT（Ahead-Of-Time Compilation）编译方式不同，后者在程序运行前就已经将源代码完全编译成机器码。

JIT编译器的优势在于能够结合解释执行和静态编译的好处：它可以在运行时进行优化，根据程序的实际执行情况来生成更高效的机器码。同时，由于JIT编译器只编译程序中实际要执行的部分，因此可以减少初次启动时间，并避免编译那些在运行过程中从未使用到的代码。

Java虚拟机（JVM）是使用JIT编译技术的一个著名例子。在JVM中，Java程序首先被编译成平台无关的字节码，随后在运行时，JIT编译器会将热点代码（经常执行的代码）编译成针对具体硬件平台优化的机器码，以提升性能。

[其他支持JIT编译的语言和运行环境包括.NET](http://xn--jit-5q9d13js0cgyd9wllkz2fa19u3w4ciyhuj5aw42ajyf2ripoa232d.net/) Framework的CLR（Common Language Runtime），JavaScript的各种现代引擎（如V8引擎）等。通过JIT编译，这些环境能够提供既快速又灵活的执行策略。

# LLVM Backend

## *指令选择*

## *寄存器分配*

## *指令调度*

## *代码生成*