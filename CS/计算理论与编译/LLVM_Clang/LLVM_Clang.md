

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
$ cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DLLVM_ENABLE_PROJECTS="clang;lldb"
# $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ../llvm
$ cmake --build build # cmake build
$ sudo cmake --build build --target install # cmake install
```

注意安装时文件可能不全，/usr/local/include/llvm/Config/config.h不会被安装进去，需要手动从build文件夹内复制出来。在运行编译时可以根据缺少头文件自行`sudo cp`

### Shell Script

```shell
# Download:
git clone https://github.com/llvm/llvm-project.git
cd llvm-project/llvm

# Configure
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release

# build:
cmake --build . -j $(nproc)

# install:
cmake -DCMAKE_INSTALL_PREFIX=${HOME}/.local/ -P cmake_install.cmake

# uninstall
xargs rm -rf < install_manifest.txt
```

### 同时安装多个版本的LLVM



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

- **LLVM_ENABLE_PROJECTS:STRING** 这个变量控制哪些项目被启用。如果只想要构建特定的LLVM子项目，比如Clang或者LLDB，可以使用这个变量来指定。例如，如果想同时构建Clang和LLDB，可以在CMake命令中加入 `-DLLVM_ENABLE_PROJECTS="clang;lldb"`
- **LLVM_ENABLE_RUNTIMES:STRING** 这个变量让能够控制哪些运行时库被启用。如果想要构建libc++或者libc++abi这样的库，可以使用这个变量。例如，为了同时构建libc++和libc++abi，应该在CMake命令中添加 `-DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi"`
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

## *卸载LLVM*

# 预定义宏

## *`__attribute__`*

和GCC一样，Clang同样支持用 `__attribute__` 来显式地控制编译器的行为，而且还添加了一些自己的扩展。为了检查一个特殊属性的可用性，可以使用`__has_attribute`指令

这里介绍一下Clang独有的 `__attribute__` 指令

### availability

[__attribute__详解及应用 | roy's blog (woshiccm.github.io)](https://woshiccm.github.io/posts/__attribute__详解及应用/)

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

# Clang 架构

[Clang C Language Family Frontend for LLVM](https://clang.llvm.org/)

[Welcome to Clang's documentation! — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/)

[Clang Compiler User’s Manual — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/UsersManual.html)

设计手册：[“Clang” CFE Internals Manual — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/InternalsManual.html)

Doxygen: [clang: clang (llvm.org)](https://clang.llvm.org/doxygen/)

> The Clang project provides a language front-end and tooling infrastructure for languages in the C language family (C, C++, Objective C/C++, OpenCL, CUDA, and RenderScript) for the [LLVM](https://www.llvm.org/) project. Both a GCC-compatible compiler driver (`clang`) and an MSVC-compatible compiler driver (`clang-cl.exe`) are provided. You can [get and build](https://clang.llvm.org/get_started.html) the source today.

## *Clang Driver*

平常使用的可执行文件 `clang.exe` 只是一个Driver，即一个命令解析器，**用于接收gcc兼容的参数**（`clang++.exe`/`clang-cl.exe`同理，用于g++/msvc兼容的参数），然后传递给真正的clang编译器前端，也就是CC1。CC1作为前端，负责解析C++源码为语法树，转换到LLVM IR。比如选项A在gcc中默认开启，但是clang规则中是默认不开启的，那么为了兼容gcc，clang.exe的Driver就要手动开启选项A，也就是添加命令行参数，将它传递给CC1

可以把Driver理解为以Clang为前端的LLVM整个软件的main函数，在这个main函数中依次调用整个编译流程中的各个阶段

### Procedure

[Driver Design & Internals — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/DriverInternals.html)

<img src="DriverArchitecture.png">

上图中橙色的代表数据，绿色代表操作这些数据的阶段，蓝色的代表辅助组件

1. Input Strings

   调用Clang的时候后面命令行传入的命令

2. Parse: Option Parsing

   将传入的String类型输入解析成具体的参数对象，如果要查看完整的解析过程，可以使用 `-###`，下面有说明

3. Pipeline: Compilation Action Construction

   一旦解析了参数，就会构造出后续编译所需要的子任务。这涉及到确定输入文件及其类型，要对它们做哪些工作（预处理、编译、组装、链接等），以及为每个任务构造一个Action实例列表。其结果是一个由一个或多个顶层Action组成的列表，每个Action通常对应一个单一的输出（例如，一个对象或链接的可执行文件）。可以使用 `-ccc-print-phases` 可以打印出这个阶段的内容

4. Bind: Tool & Filename Selection：

   这个阶段和后面的Trasnlate一起将将Actions转化成真正的进程。Driver自上而下匹配，将Actioins分配给分配给Tools，ToolChain负责为每个Action选择合适的Tool，一旦选择了Tool，Driver就会与Tool交互，看它是否能够匹配更多的Action

   一旦所有的Action都选择了Tool，Driver就会决定如何连接工具（例如，使用进程内模块、管道、临时文件或用户提供的文件名）

   Driver驱动程序与ToolChain交互，以执行Tool的绑定。ToolChain包含了特定架构、平台和操作系统编译所需的所有工具的信息，一次编译过程中，单个Driver调用可能会查询多个ToolChain，以便与不同架构的工具进行交互

   可以通过`-ccc-print-bindings` 可以查看Bind的大致情况，以下展示了在i386和ppc上编译t0.c文件Bing过程

5. Translate: Tool Specific Argument Translation

   一旦选择了一个Tool来执行一个特定的Action，该Tool必须构建具体的Commands，并在编译过程中执行。该阶段主要的工作是将gcc风格的命令行选项翻译成子进程所期望的任何选项

   这个阶段的结果是一些列将要执行Commands（包含执行路径和参数字符）

6. Execute

   执行阶段，Clang Driver 会创建两个子线程来分别之前上一阶段输出的编译和链接任务，并且产出结果

### `-###` option

```cmd
$ clang -### test.cc
clang version 19.0.0git (git@github.com:llvm/llvm-project.git 424188abe4956d51c852668d206dfc9919290fbf)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /usr/local/bin
 "/usr/local/bin/clang-19" "-cc1" "-triple" "x86_64-unknown-linux-gnu" "-emit-obj" "-dumpdir" "a-" "-disable-free" "-clear-ast-before-backend" "-disable-llvm-verifier" "-discard-value-names" "-main-file-name" "test.cc" "-mrelocation-model" "pic" "-pic-level" "2" "-pic-is-pie" "-mframe-pointer=all" "-fmath-errno" "-ffp-contract=on" "-fno-rounding-math" "-mconstructor-aliases" "-funwind-tables=2" "-target-cpu" "x86-64" "-tune-cpu" "generic" "-debugger-tuning=gdb" "-fdebug-compilation-dir=/home/wjfeng/clang_learn" "-fcoverage-compilation-dir=/home/wjfeng/clang_learn" "-resource-dir" "/usr/local/lib/clang/19" "-internal-isystem" "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11" "-internal-isystem" "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/x86_64-linux-gnu/c++/11" "-internal-isystem" "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/backward" "-internal-isystem" "/usr/local/lib/clang/19/include" "-internal-isystem" "/usr/local/include" "-internal-isystem" "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../x86_64-linux-gnu/include" "-internal-externc-isystem" "/usr/include/x86_64-linux-gnu" "-internal-externc-isystem" "/include" "-internal-externc-isystem" "/usr/include" "-fdeprecated-macro" "-ferror-limit" "19" "-fgnuc-version=4.2.1" "-fskip-odr-check-in-gmf" "-fcxx-exceptions" "-fexceptions" "-fcolor-diagnostics" "-faddrsig" "-D__GCC_HAVE_DWARF2_CFI_ASM=1" "-o" "/tmp/test-0f12cd.o" "-x" "c++" "test.cc"
 "/usr/bin/ld" "-z" "relro" "--hash-style=gnu" "--eh-frame-hdr" "-m" "elf_x86_64" "-pie" "-dynamic-linker" "/lib64/ld-linux-x86-64.so.2" "-o" "a.out" "/lib/x86_64-linux-gnu/Scrt1.o" "/lib/x86_64-linux-gnu/crti.o" "/usr/lib/gcc/x86_64-linux-gnu/11/crtbeginS.o" "-L/usr/lib/gcc/x86_64-linux-gnu/11" "-L/usr/lib/gcc/x86_64-linux-gnu/11/../../../../lib64" "-L/lib/x86_64-linux-gnu" "-L/lib/../lib64" "-L/usr/lib/x86_64-linux-gnu" "-L/usr/lib/../lib64" "-L/lib" "-L/usr/lib" "/tmp/test-0f12cd.o" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "/usr/lib/gcc/x86_64-linux-gnu/11/crtendS.o" "/lib/x86_64-linux-gnu/crtn.o"
```

`clang -### test.cc` 实际上并没有编译文件，而是告诉 Clang 输出它将会执行的命令来编译 `test.cc` 源文件。**`-###` 选项使得 Clang 打印出详细的命令行调用信息，但不真正运行这些命令**。这非常有用于调试和理解 Clang 内部的工作机制

输出显示了 Clang Driver 组装出的将要执行的两个主要步骤：

1. 使用 cc1 编译器生成目标文件

   ```cmd
   "/usr/local/bin/clang-19" "-cc1" [一系列参数] "-o" "/tmp/test-0f12cd.o" "-x" "c++" "test.cc"
   ```

   这个命令调用了 Clang 的内部前端 (`-cc1`) 来编译 C++ 源文件 `test.cc` 并生成中间对象文件 `/tmp/test-0f12cd.o`。在这个过程中，Clang 处理了包括预处理、编译和生成 LLVM IR 等任务，并最终将其转换为目标代码

   参数包括指定目标三元组 `-triple` ("x86_64-unknown-linux-gnu")、优化设置、调试信息选项、警告级别等。这里也配置了编译路径、资源目录以及系统库的包含路径等信息

2. 使用ld链接器，链接生成可执行文件

   ```cmd
   "/usr/bin/ld" [一系列参数] "/tmp/test-0f12cd.o" "-o" "a.out"
   ```

   在第二步中，调用系统的链接器 `ld` 来将之前生成的目标文件 `/tmp/test-0f12cd.o` 链接成最终的可执行文件，默认输出为 `a.out`。链接器还会链接其他启动和结束例程需要的目标文件（如 `crt1.o`, `crti.o`, `crtbeginS.o`, `crtendS.o`）和库文件（如 `-lgcc`, `-lgcc_s`, `-lc`）。这些文件和库提供了程序初始化、标准库支持和正确关闭程序所需的代码

   链接器的参数还设定了一些链接选项，比如 PIE（Position Independent Executable，位置无关可执行文件），选择动态链接器以及库和搜索路径等

## *Driver 代码*

[LLVM-Driver笔记 | 香克斯 (shanks.pro)](https://shanks.pro/2020/07/14/llvm-driver/)

[clang 01. clang driver流程分析-CSDN博客](https://blog.csdn.net/qq_43566431/article/details/130689146)

在`clang/tools/driver/driver.cpp` 我们可以找到Driver的入口，其中入口逻辑都集中在clang_main之中



<img src="AST_Action.png">

注：上图的虚线框内为回调方法，表头黑体为类名

构建AST树的核心类是ParseAST(Parse the entire filespecified,notifyingthe ASTConsumer as the file is parsed),为了方便用户加入自己的actions，Clang提供了众
多的hooks

### Frontend Action

llvm-project/clang/include/clang/Frontend/FrontendOptions.h 中的 ActionKind 枚举类

## *实操：编译Pipeline*

下图是以Clang为前端的，LLVM为后端的编译器的整体架构

<img src="Clang-LLVM-compiler-architecture.png">

以下面这份代码为例

```C++
//Example.c
#include <stdio.h>
int global;
void myPrint(int param) {
    if (param == 1)
        printf("param is 1");
    for (int i = 0 ; i < 10 ; i++ ) {
        global += i;
    }
}
int main(int argc, char *argv[]) {
    int param = 1;
    myPrint(param);
    return 0;
}
```

### 编译传参

由于Clang Driver的架构设计，需要分别用 `-Xclang` 和 `-mllvm` 分别将参数传递给Clang前端和LLVM中后段

* -Xclang参数是将参数传递给Clang的CC1前端

  比如想要禁用所有LLVM Pass的运行，也就是生成无任何优化的IR，那么就要使用-disable-llvm-passes参数传递给CC1。但是这个参数并没有Clang Driver的表示形式（也就是不使用-Xclang传递给CC1），那么就需要写-Xclang -disable-llvm-passes把参数透过Clang Driver把参数传递给CC1

* -mllvm参数的作用是将参数传递给作为中后端的LLVM

  如果参数是在LLVM中后端定义的，那么直接把参数给Clang的Driver或者CC1都是不行的，需要使用-mllvm将参数跳过Clang的Driver和CC1传递到LLVM。比如想要在Pass运行完成后输出IR，那么就需要使用-mllvm --print-after-all把参数传给LLVM

### clang & clang++

和gcc & g++的不同分工一样，clang & clang++同样分别适用于编译C和C++

clang++会自动链接C++标准库，而clang则不会

```cmd
$ clang -lstdc++ main.cpp
$ clang++ main.cpp
```

### 分别编译不同的阶段

我们可以看下有下面这些阶段

```cmd
$ clang -ccc-print-phases main.cc
            +- 0: input, "main.cc", c++
         +- 1: preprocessor, {0}, c++-cpp-output
      +- 2: compiler, {1}, ir
   +- 3: backend, {2}, assembler
+- 4: assembler, {3}, object
5: linker, {4}, image
```

1. 预处理

   ```cmd
   $ clang -E source.c -o preprocessed.i
   ```

2. 编译

   ```cmd
   $ clang -S -emit-llvm source.c -o intermediate.ll
   ```

3. 生成目标代码

   ```cmd
   $ clang -S source.c -o assembly.s
   ```

4. 汇编

   ```cmd
   $ clang -c source.c -o object.o
   ```

5. 编译

   ```cmd
   $ clang object.o -o executable
   ```

# Clang Lexer & Parser

本章介绍Clang的lexer & parser的实现

## *Lexer*

## *Parser*

Clang使用的Parser是基于递归下降分析 recursive descent parser 的

# Clang AST

[LLVM 编译器前端 Clang AST & API 学习笔记 | jywhy6's blog](https://blog.jywhy6.zone/2020/11/27/clang-notes/).











一个翻译单元 (Translation Unit) 的顶层节点是 `TranslationUnitDecl` 





## *AST 架构*

### Reminder: Expression & Statement

* Expression 表达式

  表达式是计算机程序中的一个单元，**它会计算并返回一个值**。表达式由操作数（常量、变量、函数调用等）和操作符（比如加减乘除）组成。表达式总是产生或返回一个结果值，并且可以出现在任何需要值的地方

  ```C
  3 * 7      // 返回 21 的表达式
  x          // 如果 x 是一个变量，这是一个返回 x 值的表达式
  foo()      // foo 函数调用是一个表达式；假设 foo 返回一个值
  x + y * 2  // 返回 x 与 y 乘以 2 之和的表达式
  ```

* Statement 语句

  **语句是执行特定操作的最小独立单元**，它表示要做的一项动作或命令。**语句不像表达式那样返回值**，但它可以改变程序的状态。一个程序通常由一系列顺序执行的语句构成

  ```C
  int x = 10;   // 赋值语句
  if (x > 0) {  // 条件语句
      x = -x;   
  }
  return x;     // 返回语句
  ```

  每一个语句都可以改变程序的状态，比如通过赋值语句改变变量的值，或通过条件语句来控制程序流程的走向

Expression & Statement 的主要区别在于，表达式是有返回值的，而语句可能没有。在很多语言里，语句不能作为表达式的一部分出现，因为它不返回值，而表达式可以作为更大的表达式的一部分出现，也可以独立作为表达式语句。在某些语言中（如Python），表达式可以作为语句使用，而在其他语言中（如C, C++），则必须显式使用一个表达式语句（例如，一个以分号结束的表达式）

还有一些语言（如Scala和Ruby）模糊了表达式和语句的界限，因为在这些语言中几乎所有东西都是表达式（即都有返回值）

### 核心基本类型（AST Nodes）

Clang的AST节点的最顶级类 Decl、Stmt 和 Type 被建模为没有公共祖先的独立类

* Decl 表示各种声明
  * FunctionDecl 函数声明。注意：在AST层级中，**不区分函数声明和函数定义，统一用FunctionDecl来标识**，两个区分主要看是否有函数体 function body，可以使用 `bool hasBody()` 来进行判断
  * VarDecl 变量声明，如果有初始化，可以通过 `getInit()` 获取到对应的初始化Expr
* Stmt 表示各种语句（代码块）
  * CompoundStmt 复合语句：代表大括号，函数实现、struct、enum、for的body等一般用此包起来
  * DeclStmt 定义语句，里边可能有VarDecl等类型的定义
  * ForStmt For语句对应，包括Init/Cond/Inc 对应 `(int a=0;a<mm;a++)` 这三部分，还有一部分是body，可以分别使用 `getInit()`，`getCond()`，`getInc()`，`getBody()` 来分别进行获取
  * IfStmt If语句：包括三部分Cond、TrueBody、FalseBody三部分，分别可以通过 `getCond()`，`getThen()`, `getElse()` 三部分获取，Cond和Then是必须要有的，Else可能为空
  * ReturnStmt 可选的return语句
  * ValueStmt 可能含有 Value & Type 的语句
    * Expr 表达式，clang中expression也是statement的一种
      * BinaryOperator 二元运算符
      * CallExpr 函数调用表达式，子节点有调用的参数列表
      * CastExpr 类型转换表达式
        * ImplicitCastExpr 隐形转换表达式，在左右值转换和函数调用等各个方面都会用到
      * IntegerLiteral 定点Integer值
      * ParenExpr 括号表达式
      * UnartOperator 一元操作符
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

## *实例*

[PowerPoint 프레젠테이션 (kaist.ac.kr)](https://swtv.kaist.ac.kr/courses/cs492-fall18/part1-coverage/lec7-Clang-tutorial.pdf)

```C++
//Example.c
#include <stdio.h>
int global;
void myPrint(int param) {
    if (param == 1)
    printf("param is 1");
    for (int i = 0 ; i < 10 ; i++ ) {
        global += i;
    }
}
int main(int argc, char *argv[]) {
    int param = 1;
    myPrint(param);
    return 0;
}
```

```cmd
$ clang -Xclang -ast-dump -fsyntax-only test.cc
```

- `-Xclang`：这个选项后面跟随的参数会直接传递给 Clang 的前端而不是驱动程序。Clang 驱动程序负责处理用户级别的编译选项，并将它们转化为针对各种工具（例如前端、汇编器和链接器）的实际命令行参数。使用 `-Xclang` 可以直接向 Clang 前端发送指令
- `-ast-dump`：这是传递给 Clang 前端的参数，告诉它输出 AST 的结构信息。AST 是源代码的树形表示，其中每个节点都代表了源代码中的构造（如表达式、声明等）
- `-fsyntax-only`：这个选项告诉 Clang 仅执行语法检查，而不进行代码生成或其他后续步骤。因此，它只会解析源代码，检查语法错误，并在完成后停止。这通常用于快速检查代码是否正确，或者像在这个命令中一样，与 `-ast-dump` 结合来查看源代码的 AST
-  `-fmodules` 选项启用了 Clang 的模块功能。模块是一种用于替代传统的 `#include` 预处理器指令和头文件的编译单元，它旨在改进 C 和 C++ 程序的编译时间和封装性



## *libclang*

[Libclang tutorial — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/LibClang.html)

llibclang是Clang的C语言接口库，它提供了一个相对较小的API，暴露了用于解析源代码成为AST、加载已经解析的AST、遍历AST、将物理源位置和AST中的元素关联起来以及其他支持基于Clang的开发工具的功能。这个Clang的C语言接口永远不会提供存储在Clang的C++ AST中的所有信息表示，也不应该提供：其意图是保持一个从一个版本到下一个版本相对稳定的API，只提供支持开发工具所需的基本功能

libclang的整个C语言接口可以在llvm-project/clang/include/clang-cIndex.h文件中找到

### libclang核心数据结构

**libclang中所有类型都以 `CX` 开头**

* CXIndex：一个Index包含了一系列会被链接到一起形成一个可执行文件或库的translation unit

* CXTranslationUnit：

* CXCursor：一个cursor代表了一个指向某个translation unit的AST中的某些元素

  ```C++
  typedef struct {
    enum CXCursorKind kind;
    int xdata;
    const void *data[3];	
  } CXCursor;





前序遍历

```C++
clang_visitChildren (CXCursor parent, CXCursorVisitor visitor, CXClientData client_data);
```





### Embed libclang with CMake

```cmd
$ clang++ -lcang main.cpp
```



### Python API

[libclang · PyPI](https://pypi.org/project/libclang/)



## *Traversing through AST*

Clang 主要提供了 2 种对 AST 进行访问的类：`RecursiveASTVisitor` 和 `ASTMatcher`





```C++
void clang::ParseAST (Preprocessor &pp, ASTConsumer *C, ASTContext &Ctx, bool PrintStats=false,
                      TranslationUnitKind TUKind=TU_Complete,
                      CodeCompleteConsumer *CompletionConsumer=nullptr,
                      bool SkipFunctionBodies=false);
```











### RecursiveASTVisitor

[How to write RecursiveASTVisitor based ASTFrontendActions. — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/RAVFrontendAction.html)





继承RecursiveASTVisitor，并且实现其中的 VisitCXXRecordDecl，那么这个方法就会在访问 CXXRecordDecl类型的节点上触发

### ASTMatcher

[Tutorial for building tools using LibTooling and LibASTMatchers — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/LibASTMatchersTutorial.html)





```C++
```



## *AST可视化*

[FraMuCoder/PyClASVi: Python Clang AST Viewer (github.com)](https://github.com/FraMuCoder/PyClASVi)

[CAST-projects/Clang-ast-viewer: Clang AST viewer (github.com)](https://github.com/CAST-projects/Clang-ast-viewer)

1. **Clang AST Viewer (Web Based)** 这是一个基于 Web 的工具，可以将 Clang 的 `-ast-dump` 输出转换为易于浏览的树形结构。用户可以在浏览器中直接查看以及交互式地探索 AST。
2. **Clang AST Explorer (Online Tool)** Clang AST Explorer 是一个在线工具，允许用户在网页上写代码，并实时看到对应的 AST。这个资源非常适合教学和演示目的。

# Clang Static Analyzer

[Clang Static Analyzer — Clang 19.0.0git documentation (llvm.org)](https://clang.llvm.org/docs/ClangStaticAnalyzer.html)

Clang Static Analyzer，下面简称CSA，是LLVM提供的静态分析工具

[Clang Static Analyzer 介绍 | jywhy6's blog](https://blog.jywhy6.zone/2021/05/31/clang-static-analyzer-intro/)

## *CSA中用到的数据结构*

### CSA流程

<img src="CSA流程.drawio.png">

1. CSA以源代码为起点，将源代码转换为AST
2. 将AST转换为控制流图 CFG
3. 随着程序的模拟执行，Clang 的符号执行引擎会生成 Exploded Graph 扩展图，详细记录程序的执行位置和程序当前状态信息
4. 最后，在各个 Checker（CSA中可自定义的漏洞检查器）回调函数检测到漏洞产生时，将基于 Exploded Graph 中的数据生成带漏洞触发路径的漏洞报告

### Exploded Graph

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



- `@` 全局标识
- `%` 局部标识
- `alloca` 开辟空间
- `align` 内存对齐
- `i32` 32个bit, 4个字节
- `store` 写入内存
- `load` 读取数据
- `call` 调用函数
- `ret` 返回



LLVM的优化级别分别是-O0 -O1 -O2 -O3 -Os（第一个是大写英文字母O）

Debug情况下默认是不优化，Release情况下默认Fastest、Smallest

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



# 其他LLVM工具

## *LLDB*

LLDB的使用可以看 *IDE与调试工具.md*

## *TableGen*

ableGen是LLVM项目用来定义和生成各种数据表和程序结构的一种工具。这些`.td` 文件通常包含着描述编译器组件如指令集架构、寄存器信息、指令选择规则等重要信息的声明

### TableGen工具

LLVM的TableGen工具可以从这些定义文件中生成C++代码、文档或其他格式的数据。例如，它可以被用来自动化以下任务：

- **生成寄存器描述**：TableGen可用于定义处理器的寄存器类、寄存器别名以及其他与寄存器相关的属性。
- **指令编码解码**：可以定义指令的二进制编码格式，并由此生成编码和解码指令所需的代码。
- **指令选择规则**：后端编译器的负责将中间表示转换为目标机器代码的指令选择阶段可以通过`.td`文件中的模式来定义。
- **调度信息**：给出CPU的管线模型和指令的延迟，调度算法需要此信息来进行指令重排序以提高性能。

### DSL: TableGen语言

[StormQ's Blog (csstormq.github.io)](https://csstormq.github.io/blog/LLVM 之后端篇（1）：零基础快速入门 TableGen)

### `.td` 文件内容

一个`.td`文件会包含一个或多个通过TableGen语言攥写的记录（record）格式定义的条目。这些记录描述了各种属性和值，然后被TableGen工具处理和转换。下面是一个简单的例子：

```llvm
// InstrInfo.td - Example instruction definitions for an imaginary target.

def MyTargetInst : Instruction {
  let Namespace = "MyTarget";
  bit<5> Opcode;
}

def ADD : MyTargetInst<"add", "Add two values">,
          InOperandList<[GPR, GPR]>, OutOperandList<[GPR]> {
  let Inst{31-27} = Opcode;
  let ParserMatchClass = AddRegReg;
}
```

上面的例子中，我们首先定义了一个指令类`MyTargetInst`，它有一个5位的操作码字段`Opcode`。接着我们使用该类来定义了一个加法指令`ADD`，并且指定了其输入和输出操作数列表，以及如何在解析器中匹配该指令。

最终，TableGen工具会读取`.td`文件并根据其中的定义来生成相应的代码或数据，这样开发者就不再需要手动编写大量重复而容易出错的代码了。在LLVM中，这种自动化的方法使得支持新的指令集架构或修改现有的指令集变得更加灵活和简单。
