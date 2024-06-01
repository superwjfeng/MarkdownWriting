LLVM最初是“Low Level Virtual Machine”的缩写，但随着项目的发展，它已经变成了一个品牌名称，代表整个项目本身而不特指其含义。LLVM是一种编译器基础设施，用于优化在编译时间、链接时间、运行时间和“闲置时间”进行的任何语言的任何架构的程序。LLVM提供了一套中间表示（Intermediate Representation，IR）和丰富的库，支持编译器前端和后端的开发



Clang是LLVM项目的一部分，是一个C、C++、Objective-C和Objective-C++编程语言的编译器前端。Clang使用LLVM作为其后端，能够将源码转换成LLVM IR，然后利用LLVM的优化器和代码生成器产生高效的机器码









### LLVM Compiler Infrastructure

LLVM, Low Level Virtual Machine 是一个开源的编译器基础设施项目，旨在为各种编程语言提供优化的编译器和工具，用来开发编译器前端前端和后端。LLVM的设计目标是提供可移植、高效和灵活的编译解决方案

LLVM最早以C/C++为实作对象，到目前它已支援包括ActionScript、Ada、D语言、Fortran、GLSL、Haskell、Java字节码、Objective-C、Swift、Python、Ruby、Crystal、Rust、Scala以及C#等语言

LLVM 项目由一系列模块组成，包括前端、优化器和后端。以下是 LLVM 的关键组件

<img src="LLVM的三阶段设计.png">

1. 前端（Frontend）：LLVM 前端是与特定编程语言相关的部分。它能够将不同的源代码语言转换为 LLVM 的中间表示（LLVM IR），这种中间表示是一种低级别的、面向对象的指令集表示形式，类似于汇编语言
2. 优化器（Optimizer）：LLVM 优化器是 LLVM 框架的核心组件之一。它可以对 LLVM IR 进行各种优化，包括常量折叠、循环优化、内联函数、代码消除、死代码消除等。这些优化可以显著提高程序的性能和执行效率
3. 后端（Backend）：LLVM 后端负责将优化后的 LLVM IR 转换为目标平台的机器码。LLVM 支持多种不同的目标体系结构，包括x86、ARM、MIPS等，因此可以在多个平台上生成高效的机器码
4. 工具链和库：LLVM 提供了一整套工具和库，用于构建编译器和开发工具。这些工具包括llvm-as（将汇编代码转换为 LLVM IR）、llvm-dis（将 LLVM IR 转换为可读的汇编代码）、llvm-link（将多个 LLVM 模块链接在一起）

<img src="LLVM程序分析.drawio.png">

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

     ```
     bashCopy code
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