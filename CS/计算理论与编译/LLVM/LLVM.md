# LLVM IR/LLVM Assembly

[A Gentle Introduction to LLVM IR · mcyoung](https://mcyoung.xyz/2023/08/01/llvm-ir/)

[StormQ's Blog (csstormq.github.io)](https://csstormq.github.io/blog/LLVM 之 IR 篇（1）：零基础快速入门 LLVM IR)

LLVM 可以被认为是 Clang 的优化器和后端，这可以被认为是 “LLVM 语言” 或 “LLVM 汇编” 的编译器。Clang 和其他语言前端（如 Rust）本质上编译为 LLVM IR，然后 LLVM 将其编译为机器代码

LVM IR 是LLVM增强性、优化性和灵活性的核心所在。在编译过程中，源代码首先被转换成LLVM IR，然后经过各种优化和变换处理，最终生成目标平台的机器代码。LLVM的模块化和可扩展特性使得开发者可以很容易地实现新的优化、目标后端、新语言前端并通过共享的LLVM IR进行交互。这使得LLVM成为现代编译器构建的标准工具之一

## *LLVM IR 的格式*

LLVM IR 采用静态单赋值 SSA 的中间表示。它设计用于优化、分析和生成高度优化的机器码。LLVM IR有三种表现形式：内存中表示（In-memory）、二进制字节码（Bitcode）文件和人类可读的文本表示。下面主要介绍其文本表示格式，因为这是开发者最常接触到的形式

### LLVM IR 的核心结构

* 模块 Module
  - 模块是 LLVM IR 的顶层容器，包含全局变量、函数和符号表等定义
  - 通常是由一个源文件编译生成的
* 类型系统 Type System
  - 支持基本类型（如`i32`、`float`）、复合类型（如数组、结构体）、指针类型以及函数类型等
* 函数 Function：一个函数包含参数列表、返回类型和一系列基本块（Basic Blocks）
* 基本块 Basic Block
  - 基本块是一系列顺序执行的 LLVM 指令，以终止指令（如`br`或`ret`）结尾
  - 每个基本块都有一个唯一标识符（Label）
* 指令 Instructions
  - LLVM 指令以 SSA 形式组织，共享一个无限个寄存器的抽象
  - 常见指令包括`add`、`mul`、`load`、`store`、`phi`等











## *Type*

LLVM 的类型系统非常丰富，主要包括：

- **基本类型**：整数、浮点、指针、向量等
- **复合类型**：数组、结构体、函数等
- **特殊类型**：void、label、metadata 等

```llvm
; 定义一个结构体类型
%MyStruct = type { i32, [4 x i8], float* }

; 定义一个函数类型
%MyFunc = type i32 (i32, i32)

; 定义一个函数
define i32 @foo(i32 %a, i32 %b) {
  %sum = add i32 %a, %b
  ret i32 %sum
}

; 使用向量类型
%vec = <4 x i32> <i32 1, i32 2, i32 3, i32 4>
```

### 基本类型

基本类型 Primitive Types 是 LLVM 中最简单的类型，通常用于表示单个值

* 整数类型 Integer Type

  * 用于表示整数，位宽可以是任意正整数
  * 语法：`iN`，其中 `N` 是位宽

  ```llvm
  i32     ; 32 位整数
  i1      ; 1 位整数（通常用于布尔值）
  i64     ; 64 位整数
  ```

* 浮点类型 Floating-Point Type

  - 用于表示浮点数

  - 语法：`half`、`float`、`double` 等

  ```llvm
  half    ; 16 位浮点数
  float   ; 32 位浮点数
  double  ; 64 位浮点数
  ```

* 指针类型 Pointer Type

  - 用于表示内存地址

  - 语法：`T*`，其中 `T` 是任意类型

  ```llvm
  i32*    ; 指向 i32 类型的指针
  float*  ; 指向 float 类型的指针

* 向量类型 Vector Type

  - 用于表示固定长度的向量，通常用于 SIMD 操作

  - 语法：`<N x T>`，其中 `N` 是元素个数，`T` 是元素类型

    ```llvm
    <4 x i32>   ; 包含 4 个 i32 元素的向量
    <8 x float> ; 包含 8 个 float 元素的向量
    ```

### 复合类型

复合类型 Composite Types 是由其他类型组合而成的类型

* 数组类型Array Type

  - 用于表示固定长度的数组

  - 语法：`[N x T]`，其中 `N` 是数组长度，`T` 是元素类型

  ```llvm
  [10 x i32]  ; 包含 10 个 i32 元素的数组
  [5 x float] ; 包含 5 个 float 元素的数组
  ```

* 结构体类型 Struct Type

  - 用于表示一组不同类型的字段


  - 语法：`{T1, T2, ...}`，其中 `T1`、`T2` 等是字段类型

  ```llvm
  {i32, float}        ; 包含一个 i32 和一个 float 的结构体
  {i32, [4 x i8], i64} ; 包含一个 i32、一个 4 元素 i8 数组和一个 i64 的结构体
  ```

* 函数类型 Function Type

  - 用于表示函数的签名，包括参数类型和返回值类型

  - 语法：`ret_type (param1_type, param2_type, ...)`

  ```llvm
  i32 (i32, i32)      ; 接受两个 i32 参数并返回 i32 的函数
  void (float, float) ; 接受两个 float 参数并返回 void 的函数
  ```

### 特殊类型

特殊类型 Special Types 在 LLVM 中有特殊的用途

* Void 类型 Void Type

  用于表示没有值的情况，通常用于函数的返回值

  ```llvm
  void ; 无返回值
  ```

* 标签类型 Label Type

  用于表示基本块的标签

  ```llvm
  label ; 用于标记基本块
  ```

* 元类型 Metadata Type

  用于表示附加的元数据，通常用于调试或优化

  ```llvm
  metadata ; 元数据
  ```

### 其他类型

* 不透明类型 Opaque Type

  用于表示尚未定义的类型

  ```
  opaque ; 不透明类型
  ```

* 函数指针类型 Function Pointer Type

  - 用于表示指向函数的指针

  - 语法：`ret_type (param1_type, param2_type, ...)*`

    ```
    i32 (i32, i32)* ; 指向一个接受两个 i32 参数并返回 i32 的函数的指针
    ```


## *寄存器命名*

将 %-prefixed 的名称称为 register（LLVM IR 是基于 SSA 的，所以有无限个虚拟寄存器可以使用）

### 寄存器的数字命名规则

在 LLVM IR 中，寄存器（或临时值）可以使用数字名称，例如 `%0`、`%1`、`%2` 等。这些寄存器必须按顺序定义：

- 必须先定义 `%0`，然后才能定义 `%1`，接着是 `%2`，依此类推
- 这些数字命名的寄存器通常用于表示“临时结果”，即在计算过程中生成的中间值

### 隐式命名规则

LLVM IR 中有一些隐式命名规则，可能会导致数字命名的冲突：

- **函数参数的隐式命名**：如果函数的参数没有显式命名，LLVM 会自动为它们分配数字名称，从 `%0` 开始。例如：

  ```llvm
  define void @foo(i32, i32) {
    ; 参数会被隐式命名为 %0 和 %1
  }
  ```
  
  这里的两个参数会被隐式命名为 `%0` 和 `%1`
  
- **基本块的隐式命名**：如果函数的基本块 BB （即函数体）没有显式命名，LLVM 会自动为它们分配数字名称，从下一个可用的数字开始。例如：

  ```llvm
  define void @foo(i32, i32) {
    ; 第一个基本块会被隐式命名为 %2
    %2 = add i32 %0, %1
  }
  ```
  
  这里的第一个基本块会被隐式命名为 `%2`

### 命名冲突的问题

由于隐式命名的存在，可能会导致数字命名的冲突。例如：

```llvm
define void @foo(i32, i32) {
  ; 参数被隐式命名为 %0 和 %1
  %2 = add i32 %0, %1  ; 这里试图定义 %2
}
```

在这段代码中：

- 参数已经被隐式命名为 `%0` 和 `%1`
- 第一个基本块会被隐式命名为 `%2`
- 当尝试显式定义 `%2` 时，LLVM 会报错，因为 `%2` 已经被基本块的名称占用了

这种冲突会导致非常令人困惑的错误，因为开发者可能没有意识到隐式命名的存在


### 如何避免冲突

为了避免这种问题，可以采取以下措施：

- **显式命名参数**：为函数的参数显式命名，避免隐式命名占用数字

  ```llvm
  define void @foo(i32 %a, i32 %b) {
    ; 参数被显式命名为 %a 和 %b
    %sum = add i32 %a, %b  ; 可以使用 %sum 作为临时寄存器
  }
  ```
  
- **显式命名基本块**：为基本块显式命名，避免隐式命名占用数字

  ```llvm
  define void @foo(i32 %a, i32 %b) {
  entry:
    %sum = add i32 %a, %b
  }
  ```
  
- **避免过度依赖数字命名**：尽量使用有意义的名称来命名寄存器和基本块，而不是依赖数字命名



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