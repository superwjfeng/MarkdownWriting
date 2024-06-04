# Intro

[The LLVM Compiler Infrastructure Project](https://llvm.org/)

[Getting Started with LLVM Core Libraries（中文版） — Getting Started with LLVM Core Libraries 文档 (getting-started-with-llvm-core-libraries-zh-cn.readthedocs.io)](https://getting-started-with-llvm-core-libraries-zh-cn.readthedocs.io/zh-cn/latest/index.html#)

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

## *LLVM工具链*

Clang是LLVM项目的一部分，是一个C、C++、Objective-C和Objective-C++编程语言的编译器前端。Clang使用LLVM作为其后端，能够将源码转换成LLVM IR，然后利用LLVM的优化器和代码生成器产生高效的机器码

* LLVM核心库（LLVM Core libraries）：这些库提供了一个现代的源代码和目标代码无关的优化器，并支持许多流行CPU（以及一些不太常见的CPU）的代码生成。它们围绕着称为LLVM中间表示（"LLVM IR"）的良好定义的代码表示构建。LLVM核心库文档完善，若你想创造自己的语言（或将现有编译器移植）使用LLVM作为优化器和代码生成器非常容易
* LLDB：LLDB项目基于LLVM和Clang提供的库，提供了一个出色的本地调试器。它使用Clang的AST和表达式解析器、LLVM的即时编译（JIT）、LLVM反汇编器等，从而提供了一个“开箱即用”的体验。它运行速度非常快，并且在加载符号方面比GDB更加高效
* libc++ 和 libc++ ABI：这两个项目提供了一个符合标准且高性能的C++标准库实现，包括对C++11和C++14的完全支持
* compiler-rt：compiler-rt项目提供了精调的底层代码生成支持例程的实现，如"`__fixunsdfdi`"等，当目标没有简短的本地指令序列来实现核心IR操作时会生成这些调用。它还提供了动态测试工具（例如AddressSanitizer、ThreadSanitizer、MemorySanitizer和DataFlowSanitizer）的运行时库的实现
* MLIR：MLIR子项目是构建可复用和可扩展编译器基础设施的新方法。MLIR旨在解决软件碎片化问题，改进异构硬件的编译效率，显著降低构建特定领域编译器的成本，并帮助连接已有的编译器
* OpenMP：OpenMP子项目提供了一个运行时库，用于支持Clang中实现的OpenMP
* Polly：Polly项目实现了一系列缓存局部性优化以及使用多面体模型的自动并行化和向量化
* libclc：libclc项目旨在实现OpenCL标准库
* klee：klee项目实现了一个“符号虚拟机”，该项目使用定理证明器尝试评估程序中所有动态路径，以便发现bug并证明函数的属性。klee的一个主要特点是，如果检测到bug，它可以产生一个测试用例
* LLD：LLD项目是一个新的链接器，它可以替换系统链接器，并运行得更快。
* BOLT：BOLT项目是一个链接后优化器。它通过根据采样分析器收集的执行配置文件优化应用程序的代码布局来实现性能提升

## *安装LLVM*

[LLVM Debian/Ubuntu packages](https://apt.llvm.org/)

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