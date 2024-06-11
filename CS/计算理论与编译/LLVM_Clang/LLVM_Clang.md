

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

## *LLVM工具链*

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

## *安装LLVM*

截止到2024.6.11，LLVM的最新版本为18.1.6

[Getting Started with the LLVM System — LLVM 19.0.0git documentation](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)

### 从官网上下载预编译的包

[LLVM Debian/Ubuntu packages](https://apt.llvm.org/)

### 自行编译源码

[llvm/llvm-project: The LLVM Project is a collection of modular and reusable compiler and toolchain technologies. (github.com)](https://github.com/llvm/llvm-project)



Stand-alone Builds通常指的是软件构建过程中不依赖于外部库、组件或者系统特定环境的构建方式。在这种构建方式下，软件可以单独编译和运行，不需要任何额外的软件支持。这样做的好处包括提高了软件的兼容性和移植性，因为所有必要的组件都已经包含在内，不再依赖外部环境。



### 依赖

* CMake >= 3.20.0,  Makefile/workspace generator

* python >= 3.8, Automated test suite

  Only needed if you want to run the automated test suite in the `llvm/test` directory, or if you plan to utilize any Python libraries, utilities, or bindings.

* zlib >= 1.2.3.4, Compression library

  Optional, adds compression / uncompression capabilities to selected LLVM tools.

* GNU Make 3.79, 3.79.1, Makefile/build processor

  Optional, you can use any other build tool supported by CMake.

# Example

Kaleidoscope 万花筒

# 源代码

### 源代码结构

根目录下，最重要的就是include和lib这两个文件夹。include文件夹包含了其它项目在使用LLVM核心库时需要包含的头文件，而lib文件夹里放的就是LLVM核心库的实现。分别打开lib和include，可以看到很多文件与子文件夹。有经验的读者应该能从名字大概猜到其实现的东西。比如，lib/IR子文件夹肯定是存放了与IR相关的代码，lib/Target子文件夹肯定与生成目标平台机器码有关。又比如，include/llvm/Pass.h文件里面声明了Pass类用来给你继承去遍历、修改LLVM IR。 当然，我们现在不必知道每个模块是干什么的。 等有需要再去查看官方文档吧。
根目录下还有一个tools文件夹，这里面就存放了我上面所说的周边工具。 打开这个目录，就可以看到类似llvm-as这样的子目录。显然这就是llvm-as的实现。
————————————————

                            版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。

原文链接：https://blog.csdn.net/m0_72827793/article/details/135371852



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



# LLD - The LLVM Linker









[LLD - The LLVM Linker — lld 19.0.0git documentation](https://lld.llvm.org/)





