

Doxygen of APIs: [LLVM: LLVM](https://llvm.org/doxygen/)

# Intro

[The LLVM Compiler Infrastructure Project](https://llvm.org/)

[Getting Started with LLVM Core Libraries（中文版） — Getting Started with LLVM Core Libraries 文档 (getting-started-with-llvm-core-libraries-zh-cn.readthedocs.io)](https://getting-started-with-llvm-core-libraries-zh-cn.readthedocs.io/zh-cn/latest/index.html#)

> 它最初的编写者，是一位叫做Chris Lattner(个人主页)的大神，硕博期间研究内容就是关于编译器优化的东西，发表了很多论文，博士论文是提出一套在编译时、链接时、运行时甚至是闲置时的优化策略，与此同时，LLVM的基本思想也就被确定了，这也让他在毕业前就在编译器圈子小有名气。
>
> 而在这之前，Apple公司一直使用GCC作为编译器，后来GCC对Objective-C的语言特性支持一直不够，Apple自己开发的GCC模块又很难得到GCC委员会的合并，所以老乔不开心。等到Chris Lattner毕业时，Apple就把他招入靡下，去开发自己的编译器，所以LLVM最初受到了Apple的大力支持。
>
> 最初时，LLVM的前端是GCC，后来Apple还是立志自己开发了一套Clang出来把GCC取代了，不过现在带有Dragon Egg的GCC还是可以生成LLVM IR，也同样可以取代Clang的功能，我们也可以开发自己的前端，和LLVM后端配合起来，实现我们自定义的编程语言的编译器。
>
> 原文链接：https://blog.csdn.net/RuanJian_GC/article/details/132031490

## *LLVM Compiler Infrastructure*

LLVM, Low Level Virtual Machine 是一个开源的编译器基础设施项目，旨在为各种编程语言提供优化的编译器和工具，用来开发编译器前端前端和后端。LLVM的设计目标是提供可移植、高效和灵活的编译解决方案

LLVM最早以C/C++为实作对象，到目前它已支援包括ActionScript、Ada、D语言、Fortran、GLSL、Haskell、Java字节码、Objective-C、Swift、Python、Ruby、Crystal、Rust、Scala以及C#等语言

LLVM 项目由一系列模块组成，包括前端、优化器和后端。以下是 LLVM 的关键组件

<img src="LLVM的三阶段设计.png">

1. 前端（Frontend）：LLVM 前端是与特定编程语言相关的部分。它能够将不同的源代码语言转换为 LLVM 的中间表示（LLVM IR），这种中间表示是一种低级别的、面向对象的指令集表示形式，类似于汇编语言
2. 优化器（Optimizer）：LLVM 优化器是 LLVM 框架的核心组件之一。它可以对 LLVM IR 进行各种优化，包括常量折叠、循环优化、内联函数、代码消除、死代码消除等。这些优化可以显著提高程序的性能和执行效率
3. 后端（Backend）：LLVM 后端负责将优化后的 LLVM IR 转换为目标平台的机器码。LLVM 支持多种不同的目标体系结构，包括x86、ARM、MIPS等，因此可以在多个平台上生成高效的机器码
4. 工具链和库：LLVM 提供了一整套工具和库，用于构建编译器和开发工具。这些工具包括llvm-as（将汇编代码转换为 LLVM IR）、llvm-dis（将 LLVM IR 转换为可读的汇编代码）、llvm-link（将多个 LLVM 模块链接在一起）

<img src="LLVM程序分析.drawio.png">

## *LLVM项目的组成*

* LLVM核心库（LLVM Core libraries）：这些库提供了一个现代的源代码和目标代码无关的优化器，并支持许多流行CPU（以及一些不太常见的CPU）的代码生成。它们围绕着称为LLVM中间表示（"LLVM IR"）的良好定义的代码表示构建。LLVM核心库文档完善，若你想创造自己的语言（或将现有编译器移植）使用LLVM作为优化器和代码生成器非常容易
* Clang是LLVM原生的C/C++/Objective-C编译器前端。Clang使用LLVM作为其后端，能够将源码转换成LLVM IR，然后利用LLVM的优化器和代码生成器产生高效的机器码，Clang作为编译器前端，围绕它形成了丰富的工具生态，其中包括：
  - **libclang**：提供了一个稳定的C语言API，方便其他软件作为库来使用Clang。
  - **Clang Static Analyzer**：一个静态分析工具，可以在不执行程序的情况下检查源代码中的错误。
  - **Clang Format**：用于自动格式化C/C++/Obj-C代码，以保持统一的代码风格。
  - **Clang Tidy**：一个用于诊断和修复常见编程错误的模块化和可扩展的工具。
* LLDB项目（The LLDB Debugger）基于LLVM和Clang提供的库，提供了一个出色的本地调试器。它使用Clang的AST和表达式解析器、LLVM的即时编译（JIT）、LLVM反汇编器等，从而提供了一个“开箱即用”的体验。它运行速度非常快，并且在加载符号方面比GDB更加高效
* libc++ 和 libc++ ABI：这两个项目提供了一个符合标准且高性能的C++标准库实现，包括对C++11和C++14的完全支持
* compiler-rt：compiler-rt项目提供了精调的底层代码生成支持例程的实现，如"`__fixunsdfdi`"等，当目标没有简短的本地指令序列来实现核心IR操作时会生成这些调用。它还提供了动态测试工具（例如AddressSanitizer、ThreadSanitizer、MemorySanitizer和DataFlowSanitizer）的运行时库的实现
* MLIR：MLIR子项目是构建可复用和可扩展编译器基础设施的新方法。MLIR旨在解决软件碎片化问题，改进异构硬件的编译效率，显著降低构建特定领域编译器的成本，并帮助连接已有的编译器
* OpenMP：OpenMP子项目提供了一个运行时库，用于支持Clang中实现的OpenMP
* Polly：Polly项目实现了一系列缓存局部性优化以及使用多面体模型的自动并行化和向量化
* libclc：libclc项目旨在实现OpenCL标准库
* klee：klee项目实现了一个“符号虚拟机”，该项目使用定理证明器尝试评估程序中所有动态路径，以便发现bug并证明函数的属性。klee的一个主要特点是，如果检测到bug，它可以产生一个测试用例
* LLD：LLD项目是一个新的链接器，它可以替换系统链接器，并运行得更快。
* BOLT：BOLT项目是一个链接后优化器。它通过根据采样分析器收集的执行配置文件优化应用程序的代码布局来实现性能提升



### 源代码结构

根目录下，最重要的就是include和lib这两个文件夹。include文件夹包含了其它项目在使用LLVM核心库时需要包含的头文件，而lib文件夹里放的就是LLVM核心库的实现。分别打开lib和include，可以看到很多文件与子文件夹。有经验的读者应该能从名字大概猜到其实现的东西。比如，lib/IR子文件夹肯定是存放了与IR相关的代码，lib/Target子文件夹肯定与生成目标平台机器码有关。又比如，include/llvm/Pass.h文件里面声明了Pass类用来给你继承去遍历、修改LLVM IR。 当然，我们现在不必知道每个模块是干什么的。 等有需要再去查看官方文档吧。
根目录下还有一个tools文件夹，这里面就存放了我上面所说的周边工具。 打开这个目录，就可以看到类似llvm-as这样的子目录。显然这就是llvm-as的实现。
————————————————

                            版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。

原文链接：https://blog.csdn.net/m0_72827793/article/details/135371852

# 安装 & 编译LLVM

截止到2024.6.11，LLVM的最新版本为18.1.6

[Getting Started with the LLVM System — LLVM 19.0.0git documentation](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)

## *准备工作*

### 依赖

* **CMake >= 3.20.0**,  Makefile/workspace generator

* **python >= 3.8**, Automated test suite

  Only needed if you want to run the automated test suite in the `llvm/test` directory, or if you plan to utilize any Python libraries, utilities, or bindings.

* **zlib >= 1.2.3.4**, Compression library

  Optional, adds compression / uncompression capabilities to selected LLVM tools.

* **GNU Make 3.79, 3.79.1**, Makefile/build processor

  Optional, you can use any other build tool supported by CMake.

```C++
$ sudo apt install -y gcc g++ git cmake ninja-build
```

zlib 是一个库，没有命令行的命令



## *使用预编译二进制包*

该方法适用于系统配置不足以完成编译的计算机体验LLVM，但如果未来要进行LLVM的自定义和实验，不建议使用该方法

### 从LLVM官网下载

[Download LLVM releases](https://releases.llvm.org/)

### Linux使用发行版的包管理器

* Ubuntu

  ```cmd
   $ sudo apt-get install llvm clang
  ```

* Fedora

  ```cmd
  $ sudo yum install llvm clang
  ```

Debian和Ubuntu Linux（i386和amd64）仓库可用于下载从LLVM subversion仓库编译得到的快照。[LLVM Debian/Ubuntu packages](https://apt.llvm.org/)

## *使用CMake进行编译*

### 拉取LLVM 

拉取LLVM source code [llvm/llvm-project: The LLVM Project is a collection of modular and reusable compiler and toolchain technologies. (github.com)](https://github.com/llvm/llvm-project)

github上面的是完整的LLVM项目，频繁的拉取完整的LLVM项目开销很大，以下是减少代码拉取量的设置

* shallow-clone

  ```cmd
  $ git clone --depth 1 https://github.com/llvm/llvm-project.git
  ```

* 不拉取 user branch

  ```cmd
  $ git config --add remote.origin.fetch '^refs/heads/users/*'
  $ git config --add remote.origin.fetch '^refs/heads/revert-*'
  ```

### 编译

[Getting Started with the LLVM System — LLVM 19.0.0git documentation](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)

```cmd
$ cd llvm-project
# cmake configure
$ cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lldb"
# $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ../llvm
$ cmake --build build # cmake build
$ sudo cmake --build build --target install # cmake install
```

注意安装时文件可能不全，/usr/local/include/llvm/Config/config.h不会被安装进去，需要手动从build文件夹内复制出来。在运行编译时可以根据缺少头文件自行`sudo cp`

### CMake Cache

### Stand-alone Builds

Stand-alone Builds 独立构建允许针对系统中已有的预编译版clang或LLVM库来构建子项目

You can use the source code from a standard checkout of the llvm-project (as described above) to do stand-alone builds, but you may also build from a sparse checkout or from the tarballs available on the releases page. 

对于独立构建，必须拥有一个配置正确、能够被其他项目的独立构建所使用的LLVM安装版本。这可以是发行版提供的LLVM安装，或者可以自己构建，像这样：

```cmake
cmake -G Ninja -S path/to/llvm-project/llvm -B $builddir \
      -DLLVM_INSTALL_UTILS=ON \
      -DCMAKE_INSTALL_PREFIX=/path/to/llvm/install/prefix \
      < other options >

ninja -C $builddir install
```

## *Options & Variables for CMake*

### 常用的CMake变量

`cmake --help-variable VARIABLE_NAME` 查看CMake变量的帮助

* **CMAKE_BUILD_TYPE**:STRING

  设置 make 或 ninja 的优化等级

  | Build Type         | Optimizations | Debug Info | Assertions | Best suited for            |
  | ------------------ | ------------- | ---------- | ---------- | -------------------------- |
  | **Release**        | For Speed     | No         | No         | Users of LLVM and Clang    |
  | **Debug**          | None          | Yes        | Yes        | Developers of LLVM         |
  | **RelWithDebInfo** | For Speed     | Yes        | No         | Users that also need Debug |
  | **MinSizeRel**     | For Size      | No         | No         | When disk space matters    |

  - Optimizations make LLVM/Clang run faster, but can be an impediment for step-by-step debugging.
  - Builds with debug information can use a lot of RAM and disk space and is usually slower to run. You can improve RAM usage by using `lld`, see the [LLVM_USE_LINKER](https://llvm.org/docs/CMake.html#llvm-use-linker) option.
  - Assertions are internal checks to help you find bugs. They typically slow down LLVM and Clang when enabled, but can be useful during development. You can manually set [LLVM_ENABLE_ASSERTIONS](https://llvm.org/docs/CMake.html#llvm-enable-assertions) to override the default from CMAKE_BUILD_TYPE.

* **CMAKE_INSTALL_PREFIX**:PATH

  Path where LLVM will be installed when the “install” target is built. 默认路径是/usr/local

* **CMAKE_{C,CXX}_FLAGS**:STRING

  Extra flags to use when compiling C and C++ source files respectively.

* **CMAKE_{C,CXX}_COMPILER**:STRING

  Specify the C and C++ compilers to use. If you have multiple compilers installed, CMake might not default to the one you wish to use.

### 常用的LLVM相关变量

- **LLVM_ENABLE_PROJECTS:STRING** 这个变量控制哪些项目被启用。如果你只想要构建特定的LLVM子项目，比如Clang或者LLDB，你可以使用这个变量来指定。例如，如果你想同时构建Clang和LLDB，可以在CMake命令中加入 `-DLLVM_ENABLE_PROJECTS="clang;lldb"`
- **LLVM_ENABLE_RUNTIMES:STRING** 这个变量让你能够控制哪些运行时库被启用。如果你想要构建libc++或者libc++abi这样的库，你可以使用这个变量。例如，为了同时构建libc++和libc++abi，你应该在CMake命令中添加 `-DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi"`
- **LLVM_LIBDIR_SUFFIX:STRING** 这个变量用于附加额外的后缀到库文件的安装目录。在64位架构上，你可能希望库文件被安装在`/usr/lib64`而非`/usr/lib`，那么可以设置 `-DLLVM_LIBDIR_SUFFIX=64`
- **LLVM_PARALLEL_{COMPILE,LINK}_JOBS:STRING** 构建LLVM工具链可能会消耗大量资源，尤其是链接时。使用这些选项，当你使用Ninja生成器时，可以限制并行性。例如，为了避免内存溢出（OOM）或使用交换空间(swap)，在一台32GB内存的机器上，如果你想要限制同时只有2个链接作业，可以指定 `-G Ninja -DLLVM_PARALLEL_LINK_JOBS=2`
- **LLVM_TARGETS_TO_BUILD:STRING** 这个变量控制哪些目标架构被启用。例如，如果你只需要为你的本地目标架构（比如x86）构建LLVM，你可以使用 `-DLLVM_TARGETS_TO_BUILD=X86` 来实现
- **LLVM_USE_LINKER:STRING** 这个变量允许你覆盖系统默认的链接器。例如，如果你想使用LLD作为链接器，可以设置 `-DLLVM_USE_LINKER=lld`

## *Docker*

## *Cross-compile*

LLVM的cross-compile（交叉编译）是指在一种架构或操作系统上使用LLVM工具链来编译为在不同的目标架构或操作系统上运行的代码。简而言之，交叉编译涉及生成可在与构建环境（即你正在编译代码的机器）不同的目标环境（即代码将要运行的机器）上执行的程序。

例如，你可能在一台x86架构的Linux电脑上开发软件，但是需要为ARM架构的嵌入式设备编译这个软件。使用交叉编译，你可以创建一个专门针对ARM架构的可执行文件，尽管你的开发机器是基于x86架构的。

LLVM作为一个编译器框架支持交叉编译的特性使得它非常适合开发需要在多平台上运行的软件。提供了目标三元组（target triple）的概念——一种标识目标系统的格式，包括CPU类型、制造商和操作系统等信息，以便于交叉编译器生成正确的代码。

```
<arch><sub>-<vendor>-<sys>-<abi>
```

```cmd
❯ clang --version | grep Target
Target: x86_64-unknown-linux-gnu
```



交叉编译通常用于以下情况：

1. 编写嵌入式系统或移动设备应用程序，因为这些设备通常没有足够的资源来编译复杂的代码。
2. 构建为特定操作系统或硬件优化的软件，尤其是当开发环境与目标环境不同时。
3. 创建操作系统镜像，通常在主机系统上为其他架构的设备构建系统镜像。



# Example: Kaleidoscope Language

[My First Language Frontend with LLVM Tutorial — LLVM 19.0.0git documentation](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)

[基于 LLVM 自制编译器——序 | 楚权的世界 (chuquan.me)](https://chuquan.me/2022/07/17/compiler-for-kaleidoscope-00/)

官方给出了一个 Kaleidoscope Language 万花筒语言的构建过程来展示LLVM的使用，这门语言采用手写前端（Lexer + Parser）+ LLVM IR + LLVM 后端的结构

1. [Chapter #1: Kaleidoscope language and Lexer](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl01.html) - 这展示了我们的目标和我们想要构建的基本功能。Lexer 也是为语言构建 parser的第一部分，我们使用一个简单的C++ lexer，它易于理解
2. [Chapter #2: Implementing a Parser and AST](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl02.html)- 有了lexer，我们可以讨论解析技术和基础的AST构建。本教程描述了递归下降解析和运算符优先级解析
3. [Chapter #3: Code generation to LLVM IR](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl03.html) - AST准备好后，我们将展示生成LLVM IR有多么容易，并展示如何简单地将LLVM集成到你的项目中
4. [Chapter #4: Adding JIT and Optimizer Support](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl04.html) - LLVM的一个优点是其对JIT编译的支持，因此我们将直接深入研究，并向您展示添加JIT支持只需3行代码。后续章节将展示如何生成 `.o` 文件
5. [Chapter #5: Extending the Language: Control Flow](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl05.html) - 有了基础语言的支持，我们展示了如何用控制流操作（'if'语句和'for'循环）来扩展它。这给了我们一个讨论SSA构建和控制流的机会
6. [Chapter #6: Extending the Language: User-defined Operators](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl06.html) - 本章扩展了语言，允许用户自定义任意一元和二元操作符——并且可以分配优先级！这使我们能够将“语言”的重要部分作为库例程来构建
7. [Chapter #7: Extending the Language: Mutable Variables](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl07.html) - 本章讨论如何添加用户自定义的局部变量以及赋值操作符。这展示了在LLVM中构建SSA形式有多简单：LLVM不要求您的前端构建SSA形式就可以使用它
8. [Chapter #8: Compiling to Object Files](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl08.html) - 本章解释了如何将LLVM IR编译成对象文件，就像静态编译器所做的那样
9. [Chapter #9: Debug Information](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl09.html) - 一个真正的语言需要支持试器，因此我们添加了调试信息，允许在万花筒函数中设置断点，打印出参数变量，并调用函数
10. [Chapter #10: Conclusion and other tidbits](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl10.html) - 本章通过讨论扩展语言的方法来结束整个系列，并包括指向关于“特殊主题”的信息的指针，比如添加垃圾回收支持、异常处理、调试、对“spaghetti stacks”等的支持



## *在项目中通过CMake使用LLVM*

```cmake
cmake_minimum_required(VERSION 3.20.0)
project(SimpleProject)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set your project compile flags.
# E.g. if using the C++ header files
# you will need to enable C++11 support
# for your compiler.

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Now build our tools
add_executable(simple-tool tool.cpp)

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs support core irreader)

# Link against LLVM libraries
target_link_libraries(simple-tool ${llvm_libs})
```

# Clang Lexer & Parser

[Clang C Language Family Frontend for LLVM](https://clang.llvm.org/)

[Welcome to Clang's documentation! — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/)

[Clang Compiler User’s Manual — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/UsersManual.html)

设计手册：[“Clang” CFE Internals Manual — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/InternalsManual.html)

Doxygen: [clang: clang (llvm.org)](https://clang.llvm.org/doxygen/)

> The Clang project provides a language front-end and tooling infrastructure for languages in the C language family (C, C++, Objective C/C++, OpenCL, CUDA, and RenderScript) for the [LLVM](https://www.llvm.org/) project. Both a GCC-compatible compiler driver (`clang`) and an MSVC-compatible compiler driver (`clang-cl.exe`) are provided. You can [get and build](https://clang.llvm.org/get_started.html) the source today.

下图是以Clang为前端的，LLVM为后端的编译器的整体架构

<img src="Clang-LLVM-compiler-architecture.png">

## *Lexer*

## *Parser*

# Clang AST

[LLVM 编译器前端 Clang AST & API 学习笔记 | jywhy6's blog](https://blog.jywhy6.zone/2020/11/27/clang-notes/).





```cmd
$ clang -Xclang -ast-dump -fsyntax-only test.cc
```

- `-Xclang`: 这个选项后面跟随的参数会直接传递给 Clang 的前端而不是驱动程序。Clang 驱动程序负责处理用户级别的编译选项，并将它们转化为针对各种工（例如前端、汇编器和链接器）的实际命令行参数。使用 `-Xclang` 可以直接向 Clang 前端发送指令
- `-ast-dump`: 这是传递给 Clang 前端的参数，告诉它输出 AST 的结构信息。AST 是源代码的树形表示，其中每个节点都代表了源代码中的构造（如表达式、声明等）
- `-fsyntax-only`: 这个选项告诉 Clang 仅执行语法检查，而不进行代码生成或其他后续步骤。因此，它只会解析源代码，检查语法错误，并在完成后停止。这通常用于快速检查代码是否正确，或者像在这个命令中一样，与 `-ast-dump` 结合来查看源代码的 AST







一个翻译单元 (Translation Unit) 的顶层节点是 `TranslationUnitDecl` 



## *AST 架构*

### 核心基本类型

* Decl 声明
  * FunctionDecl 函数声明
  * VarDecl 变量声明
* Stmt 语句
  * CompoundStmt 复合语句
  * BinaryOperator 二元运算符
  * Expr 表达式
    * CallExpr 函数调用表达式
    * CastExpr 类型转换表达式
* Type 类型
  * PointerType 指针类型

### ASTContext

在一个翻译单元中，所有有关 AST 的信息都在类 `ASTContext` ，包括：

- 符号表
- `SourceManager`
- AST 的入口节点: `TranslationUnitDecl* getTranslationUnitDecl()`

### Glue Classes

* DeclContext：包含其他 `Decl` 的 `Decl` 需要继承此类
* TemplateArgument：模板参数的访问器
* NestedNameSpecifier
* QualType：Qual 是 qualifier 的意思，将 C++ 类型中的 `const` 等拆分出来，避免类型的组合爆炸问题

## *Traversing through AST*

Clang 主要提供了 2 种对 AST 进行访问的类：`RecursiveASTVisitor` 和 `ASTMatcher`

### RecursiveASTVisitor

[How to write RecursiveASTVisitor based ASTFrontendActions. — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/RAVFrontendAction.html)

### ASTMatcher

[Tutorial for building tools using LibTooling and LibASTMatchers — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/LibASTMatchersTutorial.html)

## *AST可视化*

[FraMuCoder/PyClASVi: Python Clang AST Viewer (github.com)](https://github.com/FraMuCoder/PyClASVi)

[CAST-projects/Clang-ast-viewer: Clang AST viewer (github.com)](https://github.com/CAST-projects/Clang-ast-viewer)

1. **Clang AST Viewer (Web Based)** 这是一个基于 Web 的工具，可以将 Clang 的 `-ast-dump` 输出转换为易于浏览的树形结构。用户可以在浏览器中直接查看以及交互式地探索 AST。
2. **Clang AST Explorer (Online Tool)** Clang AST Explorer 是一个在线工具，允许用户在网页上写代码，并实时看到对应的 AST。这个资源非常适合教学和演示目的。

# Clang Static Analyzer

[Clang Static Analyzer — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/ClangStaticAnalyzer.html)

Clang Static Analyzer，下面简称CSA，是LLVM提供的静态分析工具

## *Checker*

以alpha开头的checker是实验版本

- DeadStores: 检测未被使用的存储，即赋值给变量但随后未使用的值。
- MallocChecker: 检测与动态内存分配相关的问题，包括内存泄漏、双重释放等。
- NullDereference: 检测潜在的空指针解引用。
- DivideZero: 检测可能导致除零错误的情况。
- UninitializedObject: 检测可能未初始化的对象使用

## *自定义Checker*

[Checker Developer Manual (llvm.org)](https://clang-analyzer.llvm.org/checker_dev_manual.html)

# Clang工具

### Clang Analyzer & Clang Tidy

Clang Analyzer和Clang Tidy都是基于LLVM项目的开源静态代码分析工具，用于帮助开发人员发现和修复C、C++和Objective-C代码中的潜在问题和错误。它们可以作为Clang编译器的附加组件使用

* Clang Static Analyzer是基于LLVM静态分析框架的一部分，旨在检测代码中的常见编程错误、内存管理问题、并发问题等。它通过对源代码进行符号执行和路径敏感分析，构建程序的控制流图，并使用各种静态分析技术来检测可能的错误和缺陷。Clang Analyzer能够识别空指针引用、内存泄漏、使用未初始化的变量、并发问题等问题，并提供相关的警告和报告
* Clang Tidy，用于进行静态代码分析和提供代码改进建议。它使用一系列可配置的检查器来检查代码，并提供建议和修复建议来改进代码质量和可读性。Clang Tidy可以检测和修复代码规范违规、不必要的复杂性、潜在的错误使用等问题。它还支持自定义规则和插件，以满足特定项目的需求

## *clang-format*

以下是如何在Linux上安装和使用 clang-format 的步骤：

1. **安装clang-format**：

   * 在大多数Linux发行版中，你可以通过包管理器来安装

     ```
     clang-format
     ```

     。例如，在基于Debian的系统中（比如Ubuntu），可以使用以下命令：

     ```cmd
     sudo apt-get install clang-format
     ```
   
2. **配置clang-format**：

   * 为了使用Google的C++风格指南，你需要在项目根目录创建一个`.clang-format`文件，该文件指定了代码格式化的风格。

   * 你可以通过运行以下命令生成一个基于Google风格的配置文件：

     ```
     bashCopy code
     clang-format -style=google -dump-config > .clang-format
     ```

   * 这会创建一个`.clang-format`文件，里面包含了Google C++风格指南的配置。

3. **使用clang-format格式化代码**：

   * 使用

     ```
     clang-format
     ```

     格式化单个文件：

     ```
     bashCopy code
     clang-format -i your_file.cpp
     ```

   * 这会根据`.clang-format`文件中的规则格式化`your_file.cpp`。

   * 也可以对整个项目中的所有C++文件进行格式化。

4. **集成到IDE或编辑器**：

   * 许多IDE和编辑器支持`clang-format`，可以集成进去以便自动格式化代码。

5. **命令行自动化**：

   * 你还可以编写脚本，来自动化格式化整个项目中的文件。

## *clang-tidy*

`Clang-Tidy` 是一个非常强大的C/C++语言的静态代码分析工具，它可以帮助发现代码中的错误、执行风格和质量检查，以及提供一些自动修复功能。这些特性使其成为C/C++开发中提高代码质量的重要工具。下面是使用`Clang-Tidy`的基本介绍：

### 安装

`Clang-Tidy`是Clang工具集的一部分。在许多系统上，你可以通过包管理器安装它：

* 在基于Debian的系统（如Ubuntu）上，使用：

  ```
  bashCopy code
  sudo apt-get install clang-tidy
  ```

* 在macOS上，使用Homebrew：

  ```
  bashCopy code
  brew install llvm
  ```

### 基本用法

一旦安装了`Clang-Tidy`，你可以在命令行中使用它来分析你的源代码文件。基本的命令行语法是：

```cmd
$ clang-tidy [options] file [-- compile_options...]
```

其中`file`是你想要分析的源文件，`compile_options`是传递给编译器的任何额外选项。

### 检查控制

`Clang-Tidy`提供了许多可配置的检查项。要查看所有可用的检查项，可以使用：

```cmd
$ clang-tidy -list-checks
```

要运行特定的检查项，可以使用`-checks=`选项，例如：

```cmd
$ clang-tidy -checks=bugprone-*,modernize-* my_file.cpp --
```

这将只运行以`bugprone-`和`modernize-`开头的检查项。

### 集成至构建系统

对于复杂的项目，直接在命令行中运行`Clang-Tidy`可能不太方便。在这种情况下，你可以将`Clang-Tidy`集成到你的构建系统中。例如，如果你使用的是CMake，可以通过设置`CMAKE_CXX_CLANG_TIDY`变量来实现：

```cmake
set(CMAKE_CXX_CLANG_TIDY clang-tidy;--checks=*;-header-filter=.*)
```

这样，每次你构建你的项目时，`Clang-Tidy`都会自动运行。

### 自动修复

`Clang-Tidy`支持一些检查的自动修复。要应用这些修复，可以使用`-fix`选项：

```cmd
$ clang-tidy -checks=... -fix my_file.cpp --
```

### 注意事项

* 由于`Clang-Tidy`进行的是静态分析，它可能无法捕捉到所有的运行时错误。
* 有些检查可能会产生误报，所以在应用任何自动修复之前，务必仔细检查。
* `Clang-Tidy`的某些功能可能依赖于你的Clang版本。

`Clang-Tidy`是一个强大的工具，可以大大提高代码质量。然而，它最好与其他工具和实践（如代码审查、单元测试等）结合使用，以形成一个全面的代码质量保证策略。

































使用现成的代码解析库，例如`libclang`，可以让你深入分析C++代码并提取出各种复杂的信息。以下是如何使用`libclang`来提取C++源文件信息以生成YAML配置文件的大致步骤：

### 步骤 1: 安装`libclang`

首先，你需要在你的系统上安装`libclang`。这通常可以通过包管理器（如`apt`，`brew`等）或从[LLVM官方网站](http://releases.llvm.org/download.html)下载预编译的二进制文件。

例如，在Ubuntu上，可以使用：

```cmd
sudo apt-get install libclang-dev
```

### 步骤 2: 创建解析程序

创建一个C++程序或脚本，用于调用`libclang`的API并遍历AST（抽象语法树）。你可能会使用`libclang`的`CXCursor`类型和相关函数来访问代码的不同部分。

### 步骤 3: 遍历AST并提取数据

使用`clang_visitChildren`函数来遍历AST中的每个节点，并通过判断节点类型来决定是否提取信息。以下是一段简化的伪代码示例：

```C++
#include <clang-c/Index.h>
#include <iostream>
#include <string>

// 递归遍历AST并打印信息
CXChildVisitResult visitor(CXCursor cursor, CXCursor parent, CXClientData client_data) {
    // 获取当前节点的类型
    CXCursorKind kind = clang_getCursorKind(cursor);

    // 过滤感兴趣的节点类型，例如类、方法等
    if (kind == CXCursor_ClassDecl) {
        // 获取类名
        CXString className = clang_getCursorSpelling(cursor);
        std::cout << "Class: " << clang_getCString(className) << std::endl;
        clang_disposeString(className);
        // 继续向下遍历
        clang_visitChildren(cursor, visitor, nullptr);
    } else if (kind == CXCursor_CXXMethod && clang_Cursor_isPublic(cursor)) {
        // 获取函数名
        CXString methodName = clang_getCursorSpelling(cursor);
        std::cout << "Method: " << clang_getCString(methodName) << std::endl;
        clang_disposeString(methodName);
    }
    
    return CXChildVisit_Recurse;
}

int main(int argc, char **argv) {
    CXIndex index = clang_createIndex(0, 0);
    // 假设 argv[1] 是要解析的C++源文件的路径
    CXTranslationUnit unit = clang_parseTranslationUnit(index, argv[1], nullptr, 0, nullptr, 0, CXTranslationUnit_None);

    if (unit == nullptr) {
        std::cerr << "Unable to parse translation unit." << std::endl;
        exit(-1);
    }

    CXCursor rootCursor = clang_getTranslationUnitCursor(unit);
    clang_visitChildren(rootCursor, visitor, nullptr);

    clang_disposeTranslationUnit(unit);
    clang_disposeIndex(index);

    return 0;
}
```

上面的代码展示了如何使用`libclang`遍历AST并打印出类名和公有方法名。实际应用中，你需要将这些信息存储起来，以便最后输出到YAML文件中。

### 步骤 4: 输出YAML格式数据

为了将提取的信息输出为YAML格式，你可以使用`yaml-cpp`库，或者简单地手动构建YAML字符串。如果使用库，则需要先安装`yaml-cpp`，然后引入到你的项目中。

创建YAML结构并填充数据，最后写入文件：

```C++
#include <yaml-cpp/yaml.h>
// ...

// 假设已经有了一个用于存储类和方法信息的结构
std::map<std::string, std::vector<std::string>> classes;

// ...

// 在visitor函数中填充classes map
// ...

// 然后使用yaml-cpp生成YAML文件
YAML::Emitter out;
out << YAML::BeginMap;

for (const auto& pair : classes) {
    out << YAML::Key << pair.first; // 类名作为key
    out << YAML::Value << YAML::BeginSeq;
    for (const auto& method : pair.second) {
        out << method; // 方法列表
    }
    out << YAML::EndSeq;
}

out << YAML::EndMap;

// 写入文件或标准输出
std::ofstream fout("output.yaml");
fout << out.c_str();
```

完成以上步骤之后，你将能够从C++源文件中提取出所需的信息，并将其以YAML的格式保存到文件中。注意，实际情况可能更加复杂，你可能需要处理C++的高级特性，比如模板、宏、命名空间、重载函数等

# LLVM IR

IR是LLVM的核心所在

# 代码优化 Pass

## *Polly*

Polly 是 LLVM 项目的一个子项目，它提供了自动并行化和循环优化的功能。Polly 使用高级多维数组索引（Affine Expressions）来理解、表示和优化循环嵌套，特别是那些对于性能至关重要的计算密集型循环

Polly 基于一种叫做多面体模型的数学表示，使用这种方法，可以进行复杂的优化

Polly 主要应用于需要大规模数值计算的科学和工程领域，例如物理模拟、矩阵运算和图像处理。在这些领域，循环结构往往占据了程序的绝大部分计算时间，并且有明确的数据依赖模式可供分析和优化

# JIT Compiler

JIT Compiler（Just-In-Time Compiler）即时编译器，它在运行时（即程序执行期间）将程序的源代码或字节码动态地编译成机器码，然后立即执行。这与传统的AOT（Ahead-Of-Time Compilation）编译方式不同，后者在程序运行前就已经将源代码完全编译成机器码。

JIT编译器的优势在于能够结合解释执行和静态编译的好处：它可以在运行时进行优化，根据程序的实际执行情况来生成更高效的机器码。同时，由于JIT编译器只编译程序中实际要执行的部分，因此可以减少初次启动时间，并避免编译那些在运行过程中从未使用到的代码。

Java虚拟机（JVM）是使用JIT编译技术的一个著名例子。在JVM中，Java程序首先被编译成平台无关的字节码，随后在运行时，JIT编译器会将热点代码（经常执行的代码）编译成针对具体硬件平台优化的机器码，以提升性能。

[其他支持JIT编译的语言和运行环境包括.NET](http://xn--jit-5q9d13js0cgyd9wllkz2fa19u3w4ciyhuj5aw42ajyf2ripoa232d.net/) Framework的CLR（Common Language Runtime），JavaScript的各种现代引擎（如V8引擎）等。通过JIT编译，这些环境能够提供既快速又灵活的执行策略。



# 指令选择

# LLD - The LLVM Linker









[LLD - The LLVM Linker — lld 19.0.0git documentation](https://lld.llvm.org/)





