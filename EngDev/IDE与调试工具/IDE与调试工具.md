# VS

VS文档：<[Visual Studio 文档 | Microsoft Learn](https://learn.microsoft.com/zh-cn/visualstudio/windows/?view=vs-2022)>

<img src="VS中的图标.png">

## *生成配置*

### Debug和Release版本

* Debug调试版本：包括了调试信息，并且不做任何优化，便于程序员调试程序
* Release发布版本：编译器进行了各种优化，使得程序在代码大小和运行速度上都是最优的，以便用户使用，但不能进行调试

只有DEBUG版的程序才能设置断点、单步执行、使用 TRACE/ASSERT等调试输出语句。release不包含任何调试信息，所以体积小、运行速度快

对于x86来说，Debug和Release编译生成的obj和链接后可执行文件会分别放在Debug和Release文件夹中；而x86则是将Debug和Release分别凡在名为x64的文件夹中

## *项目结构*

### 如何组织一个项目？

新建的vs工程的配置文件主要包括两部分：Solution（解决方案）和Project（工程）配置文件

一个解决方案里可能包含多个工程。**每个工程是一个独立的软件模块，比如一个程序、一个代码库等。**这样的好处是解决方案可以共享文件和代码库

### 解决方案文件

VS 采用两种文件类型（`.sln` 和 `.suo`）来存储解决方案设置，下面提到的除了`.sln` 之外的文件都放在 `.vs/解决方案名/版本号/` 中

* `*.sln` Visual Studio.Solution  环境提供对项目、项目项和解决方案项在磁盘上位置的引用，可以将它们组织到解决方案中。比如是生成Debug还是Release，是通用CPU还是专用的等。`*.sln`文件可以在开发小组的开发人员之间共享。 `.sln` 就是打开文件的索引，正确引导用户进入环境、进入工程
* `*.suo`  Solution User Operation 解决方案用户选项，记录所有将与解决方案建立关联的选项，以便在每次打开时，它都包含用户所做的自定义设置。比如说VS窗口布局、项目最后编译的而又没有关掉的文件在下次打开时用，打的断点等。注意： `*.suo` 文件是用户特定的文件，不能在开发人员之间共享

sdf 和 ipch文件与VS提供的智能感知、代码恢复、团队本地仓库功能有关，如果不需要，可以禁止，就不会产生sdf 和 ipch这两个文件了，VS重新加载解决方案时速度会加快很多。另外这两个文件会导致VS工程变得很大，如果此时用git进行管理，git中的管理文件也会变得很大

* `*.sdf`文件：SQL Server Compact Edition Database File（`.sdf`）文件，是工程的信息保存成了数据库文件。sdf文件是VS用于intellisense的
  * 若没有参加大型的团队项目，不会涉及到高深的调试过程，这个文件对于用户来说是没什么用的，可以放心删除。若后来又需要这个文件了，只打开工程里的 `.sln` 文件重新编译链接就ok了
  * 同时我们注意到，当我们打开工程的时候还会产生一个 `*.opensdf` 的临时文件，不需关心，该文件关闭工程就会消失，是update `*.sdf`文件的缓冲。如果完全不需要，也觉得sdf文件太大，那么可以：在Visual Studio里进入如下设置：进入“Tools > Options”，选择“Text Editor >C/C++ > Advanced”，然后找到“Fallback Location”。然后把“Always use Fallback Location”和“Do Not Warn if Fallback Location”设置成“True”。这样每次打开工程，不会再工程目录生成 `*.sdf` 文件了
  * VS2015之后生成的 `*.db` 文件是sqlite后端用于intellisense的新数据库，相当于之前的 `*.sdf` SQL Server Compact数据库。它与VS2015提供的智能感知、代码恢复、团队本地仓库功能有关，VS重新加载解决方案时速度超快
* ipch文件夹：用来加速编译，里面存放的是precompiled headers，即预编译好了的头文件
* 禁止这两类文件生成的设置方法是：工具 `->` 选项 `->` 文本编辑器 `->` C/C++ `->` 高级，把回退位置和警告设置为true或者禁用数据库设为true，这样就不会产生那个文件了

上面的文件只是起一个组织的作用，将各个信息凝聚在一起，从而形成一个解决方案。不要随意的删掉着写看似没用的文件，删掉代码也不会丢失，但是，有时候环境配置好后，使用也比较方便，对于这两个文件，没必要动它。为了减少项目文件的大小，和这两个文件没有关系，但是如果操作不当，会导致解决方案打不开。那么解决办法就只有重建项目，然后导入代码文件了，只是会浪费一些时间而已，又要重新组织项目文件

### 工程配置文件

 Project的配置文件种类主要包括：`*.vcxproj`、`*.vcxproj.filters`、`*.vcxproj.user`、`*.props`.（注意区分`*.vcproj` 和 `*.vcxproj` 的区别，前者是vs2008及以前版本的工程配置文件，后者是vs2010及以后的工程配置文件）

* `*.vcxproj`文件是真正的项目配置文件，以**标准XML格式**的形式记录了工程的所有配置，如包含的文件名、定义的宏、包含的头文件地址、包含的库名称和地址、系统的种类等等。此外，还可以使用过滤条件决定配置是否有效
* `*.vcxproj.filters` 文件是项目下文件的虚拟目录，用于组织项目中源代码文件的视图层次结构的XML文件。它定义了在 Visual Studio 中的解决方案资源管理器中如何显示和组织项目文件。该文件通常包含项目中的文件夹结构和源代码文件的过滤器（例如，源文件、头文件、资源文件等）。通过在`*.vcxproj.filters`文件中定义过滤器，可以在 Visual Studio 中更好地组织和浏览项目文件
* `*.vcxproj.user` 是XML格式的用户配置文件，用于保存用户个人的数据，比如配置debug的环境PATH等等。用于存储针对特定用户的项目设置。这些设置通常包括编译器选项、调试器设置、运行时环境等。每个用户在打开或修改项目时，可以在该文件中保存自己的首选项和个性化设置
* `*.props` 是属性表文件，用于保存一些配置，可以根据需求，导入到项目中使用。使用起来很灵活，比如使用一个开源库，我们新建一个工程，往往需要做不少配置，如果不用属性表文件的话，那么我们每次创建一个工程都要配置一遍，太浪费时间了。如果我们能将配置保存起来，每次新建项目将配置加进来就好了，属性表文件就很好的实现了这一点

## *IntelliSence*

IntelliSense 是一项由 Microsoft 开发的智能代码补全和代码提示功能，旨在提高开发人员在集成开发环境（IDE）中编写代码的效率和准确性。它在多个 Microsoft IDE（如 Visual Studio、Visual Studio Code）和其他编辑器中得到广泛支持。

IntelliSense 使用静态代码分析、语义分析和用户输入上下文来为开发人员提供有关代码的实时信息和建议。它的主要功能包括：

1. 代码自动补全：IntelliSense 会根据正在输入的代码上下文，提供相关的代码补全选项。它可以自动完成代码片段、类、函数、变量等，并显示对应的参数列表和函数签名。
2. 代码导航：IntelliSense 可以帮助开发人员快速浏览代码库，并提供与代码相关的导航功能。这包括跳转到定义、查看函数调用层次结构、查找引用等。
3. 实时错误检查：IntelliSense 可以在代码编写过程中进行实时的语法和语义错误检查，并显示相应的错误和警告。这样可以帮助开发人员及早发现和修复问题，提高代码质量。
4. 文档注释：IntelliSense 可以显示与代码相关的文档注释、函数说明和参数描述，使开发人员能够更好地理解代码的含义和使用方式。
5. 提示和上下文帮助：IntelliSense 可以根据用户输入的上下文，提供有关可用选项的提示和帮助。它可以显示函数签名、参数类型、属性和方法列表等信息，以便开发人员更准确地编写代码

## *VS远程开发*

### Samba服务器

Samba 是一个开源的网络协议套件，允许不同操作系统之间共享文件和打印机。它使Linux、Unix 和类似系统可以与Windows 系统互操作，允许在不同操作系统之间共享文件和资源。以下是关于Samba服务器的一些重要信息：

1. **文件共享**：Samba允许在Linux/Unix和Windows系统之间共享文件和目录。这意味着您可以在Linux服务器上创建共享文件夹，并允许Windows用户通过网络访问这些文件夹，就好像它们位于Windows本地文件系统中一样。
2. **打印机共享**：除了文件共享，Samba还支持共享打印机。这使得Windows用户可以使用网络上的共享打印机，无需在其本地系统上安装驱动程序。
3. **支持多种协议**：Samba支持多种网络文件共享协议，包括SMB/CIFS（Server Message Block / Common Internet File System）、SMB2和SMB3等。这使得它与不同版本的Windows系统和其他操作系统兼容。
4. **安全性**：Samba提供了强大的安全性功能，可以配置访问控制列表（ACLs）和权限，以确保只有授权的用户能够访问共享资源。它还支持用户身份验证和加密来保护数据传输的安全性。
5. **域控制器**：Samba还可以用作域控制器，允许您在Linux系统上创建和管理Windows活动目录域。这使得在混合操作系统环境中实现统一的用户和资源管理变得更加容易。
6. **跨平台**：Samba是一个跨平台的解决方案，可以在多种操作系统上运行，包括Linux、Unix、BSD等。

# gdb调试器

通常在程序开始运行之前，可以使用一种特殊的方法来启动和运行这个程序，这种方法就是通过一个称为 调试器 Debugger 的工具。调试器是一种专门的软件，它允许程序员以一种特殊的方式运行程序，这样就可以更仔细地观察和检查程序的行为

在一般情况下，程序直接在操作系统中运行，而使用调试器启动程序时，程序是在调试器的控制下运行的

http://c.biancheng.net/gdb/

## *调试执行*

### 启动调试

gcc/g++编译出来的二进制程序默认是release模式，**要使用gdb调试，必须在源代码生成二进制程序的时候加上选项 `-g`**

* 两种加载调试文件的方式
  * 直接 `gdb file_name` 来加载调试文件
  * 如果是先打开了gdb，可以通过 `file file_name` 来加载调试文件 
* 退出GDB：当完成调试时，可以使用 `q` 或者 `quit` 命令退出GDB

### 附加到进程

在很多情况下，程序出现问题时并不处于调试状态。也就是说在我们想要调试程序时，程序已经加载到内存中开始运行，此时并不处于Debugger的控制下

可以使用 `gdb attach pid` 的方式把一个已经run起来的程序托管给gdb运行

## *断点管理*

### 命令类型

在gdb中使用help可以得到11种命令类型+1种用户自定义命令+1种命令别名

* aliases -- User-defined aliases of other commands. 给命令取别名
* breakpoints -- Making program stop at certain points. 断点命令
* data -- Examining data. 查看数据
* files -- Specifying and examining files. 对Debug对象文件的操作
* internals -- Maintenance commands.
* obscure -- Obscure features.
* running -- Running the program. Debug过程控制
* stack -- Examining the stack. 堆栈信息
* status -- Status inquiries.
* support -- Support facilities.
* text-user-interface -- TUI is the GDB text based interface.
* tracepoints -- Tracing of program execution without stopping the program.
* user-defined -- User-defined commands.

### 断点类型

* 行号断点：通过在源代码的特定行上设置断点，可以使程序在执行到该行时停止。`break [filename:]linenumber`
* 函数断点：通过指定要在特定函数内停止程序执行的方式来设置函数断点。`break function_name`
* 条件断点：设置一个条件，只有当条件满足时才会触发断点。`break location if condition`
* 硬件断点：硬件断点是在处理器级别实现的断点，可以用于监视内存地址的读写操作。`break location hardware`
* 监视断点：监视断点用于监视变量的值的更改。当变量的值发生变化时，程序会停止执行。`watch variable`
* 静态断点：静态断点是指在程序启动之前设置的断点，用于在程序加载时立即生效。`break filename:linenumber static`

### 断点操作

* 设置断点：在程序中设置断点，以在特定位置停止程序的执行。使用 `break` 命令，后面跟上文件名和行号或函数名
* 查看断点：`info break`
* 启用和禁用断点：使用 `enable` 和 `disable` 命令可以分别启用和禁用一个或多个断点。例如，`enable breakpoints` 或 `disable breakpoints`
* 删除断点
  * `delete` 删除所有断点；`delete 断点编号` 删除指定编号的断点；`delete 范围` 删除编号范围内的断点
  * `clear 函数名` 删除函数断点；`clear 行号` 删除指定行号的断点
* 临时断点：可以设置一个临时断点，它会在首次触发后自动删除。使用 `tbreak` 命令来设置临时断点，例如，`tbreak function_name`

### 断点的重复操作

1. **忽略计数**：
   * 可以使用 `ignore` 命令来设置一个断点的忽略计数，以指定触发断点的次数。例如，`ignore 3 1` 表示在第3次触发后停止。
2. **条件断点修改**：
   * 使用 `condition` 命令可以更改条件断点的条件。例如，`condition breakpoint_number new_condition`。

## *程序执行*

* 程序执行

  * `list`（或 `l`）：查看源代码，默认查看当前运行行的前后各5行。用 `l-` 来查看前后各10行。也可以用 `set listsize NUM` 来修改查看的行数
  * 使用 `run`（或 `r`）命令来启动程序，并在达到断点或程序结束时停止
  * 使用 `continue`（或 `c`）命令来从中断点继续程序直到下一个断点

* 运行控制

  * `step`（或 `s`）：逐语句执行程序，进入函数内部

  * `next`（或 `n`）：逐过程执行程序，跳过函数内部
  * `finish`：继续运行直到当前函数执行完成

## *查看信息*

### 查看和修改变量的值

* 查看变量的值
  * `print`（或 `p`）：查看变量的值，例如 `p variable_name`
  * `info locals`：查看当前作用域内的局部变量
  * 查看结构体/类的值：使用 `p *结构体指针` 直接查看结构体/类的整体信息
  * 查看数组
  * 监视断点可以持续监控
  *  `display` 命令可以设置一个表达式，每次程序停止时都会显示该表达式的值。这对于跟踪变量的值变化很有用
* 修改变量的值
  * 用 `set` 命令可以修改变量的值。需要指定变量名和新值

### 查看内存

* 使用 `x` 命令可以查看内存中的内容。需要指定要查看的内存地址和显示的格式

  ```
  x /[格式] [内存地址]
  ```

  格式可以是十六进制、十进制、八进制等。例如，`x/4x 0x12345678` 将以十六进制格式查看地址 `0x12345678` 处的四个字节

* 查看字符串：要查看内存中的字符串，可以使用 `x/s` 命令

   ```
   x/s [内存地址]
   ```
   
   例如，`x/s 0xabcdefg` 将以字符串格式查看地址 `0xabcdefg` 处的内容。
   
* 查看内存块：如果要查看一段连续的内存，可以使用 `x` 命令的范围选项

   ```
   x/[格式] [起始地址] [结束地址]
   ```
   
   例如，`x/16x 0x1000 0x1010` 将以十六进制格式查看地址从 `0x1000` 到 `0x1010` 的内容。
   
* 查看局部变量的内存：可以使用 `print` 命令结合 `&` 运算符来查看局部变量的内存地址，然后再使用 `x` 命令来查看该地址处的内容。

### 查看寄存器

在程序运行或已暂停的状态下，使用 `info registers` 命令查看寄存器的值。gdb将显示当前程序中所有可见寄存器的值和它们的名称

```
info registers
i r // 等价缩写
```

如果你关心特定寄存器的值，可以使用 `info register` 命令，并指定要查看的寄存器的名称。例如，要查看EAX寄存器的值，可以执行以下命令：

```
info register eax
```

### 查看调用堆栈

* `backtrace`（或 `bt`）：查看栈回溯信息
* 切换栈帧：`frame 栈帧号` 也可以通过 up 和 down 这两个命令用于在函数调用堆栈之间切换：`up` 命令将当前堆栈帧移动到调用者，而 `down` 命令将当前堆栈帧移动到被调用的函数。
* 查看帧信息：`info frame`

### 线程管理

# 转储文件调试分析

Coredump文件，即核心转储文件 是在计算机程序崩溃或发生严重错误时生成的一种文件，用于帮助开发人员诊断和调试问题。这个文件包含了程序在崩溃时内存中的状态信息，包括变量的值、函数调用堆栈和程序计数器等信息

Coredump文件相当于是程序在眼中错误发生时刻的快照。注意：Coredump文件可能包含敏感信息。确保在分析完Coredump文件后，将其删除或以安全的方式处理，以防止敏感数据泄漏

## *Linux Coredump*

### 前期设置

默认会在程序的当前目录生成coredump文件

1. 设置coredump文件生成的目录：其中 `%e` 表示程序文件名，`%p` 表示进程ID

   ```cmd
   $ echo /data/coredump/core.%e.%p > /proc/sys/kernel/core_pattern
   ```

2. 保证当前执行程序的用户对 coredump 目录有写权限且有足够的空间存储来 coredump 文件

3. 生成不受限制的 coredump 文件（默认是0）

   ```cmd
   $ ulimit -c unlimited
   ```

### 调试

```cmd
$ gdb program core_file
```

### 直接打印堆栈信息

可以使用信号的handle函数来获取栈信息，然后在日志中打印出来

```c
void handle_segv(int signum) {
	void *array[100];
    size_t size;
    char **strings;
    size_t i;
    signal(signum, SIG_DFL);
    size = backtrace(array, 100);
    strings = (char **)backtrace_symbols(array, size);
    fprintf(stderr, "Launcher received SIG: %d Stack trace:\n", signum);
    for (i = 0; i < size; i++) {
        fprintf(stderr, "%d %s\n", i, strings[i]);
    }
    free(strings);
}
```

在main函数中加入

```c
signal(SIGSEGV, handle_segv);  // SIGSEGV 11 Core Invalid memory reference
signal(SIGABRT, handle_segv);  // SIGABRT 6 Core Abort signal from
```

打印出的地址信息可以用下面的命令来转换

```cmd
$ addr2line -a <堆栈地址> -e <程序名>
```



## *Win Coredump*

# VS Code中配置开发环境

## *CMake*

### CMake Generator

CMake Generator 是 CMake 工具的一个组件，用于控制如何生成构建系统的文件。简单来说，CMake 是一个跨平台的自动化构建系统，它使用  CMakeLists.txt 定义项目的构建过程。当运行 CMake 时它读取这些文件，并根据指定的生成器生成相应的构建系统文件

生成器决定了 CMake 生成哪种类型的构建文件。比如说若使用的是 Visual Studio，CMake 可以生成 Visual Studio 解决方案和项目文件；若使用的是 Make，它可以生成 Makefile。这意味着可以在一个项目中使用相同的 CMakeLists.txt 文件，并根据需要生成不同的构建系统文件

在Ubuntu中输入 `cmake` 可以看到它支持下面的生成器

```
  Green Hills MULTI            = Generates Green Hills MULTI files
                                 (experimental, work-in-progress).
* Unix Makefiles               = Generates standard UNIX makefiles. 适用于 Unix-like 系统上的 Make 工具
  Ninja                        = Generates build.ninja files. 一个小型但非常快速的构建系统
  Ninja Multi-Config           = Generates build-<Config>.ninja files.
  Watcom WMake                 = Generates Watcom WMake makefiles.
  CodeBlocks - Ninja           = Generates CodeBlocks project files.
  CodeBlocks - Unix Makefiles  = Generates CodeBlocks project files.
  CodeLite - Ninja             = Generates CodeLite project files.
  CodeLite - Unix Makefiles    = Generates CodeLite project files.
  Eclipse CDT4 - Ninja         = Generates Eclipse CDT 4.0 project files.
  Eclipse CDT4 - Unix Makefiles= Generates Eclipse CDT 4.0 project files.
  Kate - Ninja                 = Generates Kate project files.
  Kate - Unix Makefiles        = Generates Kate project files.
  Sublime Text 2 - Ninja       = Generates Sublime Text 2 project files.
  Sublime Text 2 - Unix Makefiles
                               = Generates Sublime Text 2 project files.
```

**在Linux上使用VS Code时默认的生成器是Ninja**

选择哪个生成器通常取决于具体所使用的开发环境和平台。CMake 通过提供这种灵活性，使得开发者可以轻松地在不同的平台和工具之间移植他们的项目

### Ninja

https://ninja-build.org

Ninja是一个专注于速度的小型构建系统，它被设计用来运行与其他构建系统（如CMake）的生成规则。Ninja的主要目标是提高重建的速度，尤其是对于那些大型代码库的小的增量更改。在实践中，Ninja通常不是直接由开发人员手动使用，而是作为更高级别工具（如CMake）的一部分自动调用，以提供更快的构建时间和更高效的增量构建

以下是Ninja的一些关键特点：

* 快速性能：Ninja的核心优势在于它的速度。它通过最小化磁盘操作和重新计算依赖性来实现快速的构建时间。这对于大型项目尤其重要，其中即使很小的更改也可能触发大量的重新编译
* 简单性：Ninja的设计哲学强调简单性。它的配置文件（Ninja文件）简洁易懂。这种设计使得Ninja作为底层构建系统的理想选择，可以被更复杂的系统（如CMake）作为后端使用
* 非递归：Ninja使用非递归模型来处理构建规则，这有助于提高性能并减少复杂性
* 依赖处理：Ninja可以精确地处理依赖关系，以确保在构建过程中只重建必要的部分
* 跨平台支持：Ninja支持多种操作系统，包括Linux, Windows和macOS，这使得它成为在不同平台上进行项目构建的理想工具
* 用于大型项目：Ninja特别适合大型项目，如Chrome或Android。这些项目可以从Ninja的快速迭代和构建过程中受益

## *自动化任务*

### 新建工程

VS Code没有传统意义上的“创建新项目”流程（如在某些IDE中所见），但当打开一个文件夹 folder 时，它实际上就被视为一个工程 Project 或工作区 Workspace。VS Code会将这个文件夹及其内容作为一个整体来管理，这包括但不限于：

* **文件和目录结构**：可以在侧边栏的资源管理器中浏览和管理文件夹内的所有文件和子目录
* **设置和配置**：特定于项目的设置（例如，编译器配置、调试配置等）通常存储在文件夹内的`.vscode`目录中。这些设置包括`tasks.json` 和`launch.json` 等文件
* **扩展和依赖性**：VS Code允许用户为特定工作区安装扩展，这些扩展可以为项目提供额外的功能，如语言支持、代码格式化、版本控制等
* **版本控制集成**：如果项目使用Git或其他版本控制系统，VS Code可以集成这些功能，提供源代码管理和版本控制的便捷界面

在VS Code 中创建工程，并配置运行和调试环境涉及两个关键文件：`launch.json`（用于定义构建任务） 和 `tasks.json`（用于配制调试）。这些文件位于工程项目的 `.vscode` 文件夹中。下面介绍这两个文件

### `tasks.json` 定义构建任务

```
linting 代码校验 -> building 编译 -> packaging 打包 -> testing 测试 -> 部署 deployment
```

在软件开发中，自动化工具会帮助我们完成上面的流程。对于C/C++而言，lingting由Clang-tidy、Cppcheck来完成；编译由g++、gdb完成；编译打包测试部署由Makefile和CMake等自动化build工具来完成；test则由gtest等单元测试工具完成

`tasks.json` 则是VS Code中用于定义和配置任务 Tasks，这些任务通常用于自动化编译、运行、测试和其他与项目相关的操作。对于C++编程，`tasks.json` 文件的作用非常重要，因为它允许你在编辑器中设置和运行C++编译器以及其他自定义任务

在软件开发中，任务 task 是一个可执行的操作或工作单元，通常用于执行某种特定的操作，例如编译代码、运行测试、执行部署、清理项目等。任务可以是自动化的，可以由开发者手动触发，也可以作为自动化流程的一部分在CI/CD中运行

以下是 `tasks.json` 文件在C++编程中的主要作用：

* 定义编译任务：可以使用 `tasks.json` 来定义C++项目的编译任务，包括编译器命令、编译选项、源文件和目标文件等。这使得我们通过简单的键盘快捷方式或菜单选项来触发编译操作，而不必手动运行命令行编译器
* 配置自定义任务：除了编译任务，还可以在 `tasks.json` 中定义其他自定义任务，比如运行测试、生成文档、清理项目等。这些任务可以根据项目的需求进行定制
* 集成外部工具：可以使用 `tasks.json` 集成外部工具，例如代码生成工具、构建工具、自动化部署脚本等。通过配置任务可以在VS Code中轻松调用和执行这些外部工具
* 多平台支持：`tasks.json` 可以根据不同的操作系统和开发环境定义不同的任务配置。这使得我们可以在不同平台上构建和运行项目，而无需手动修改配置
* 任务依赖性：可以定义任务之间的依赖关系，确保它们以正确的顺序执行。例如，在构建任务之前，可能需要先运行代码校验任务
* 自动化工作流程：通过将任务与VS Code的自动保存、自动构建和自动测试功能结合使用，可以实现自动化的开发工作流程，提高开发效率

### 属性

task的总体结构如下

```json
tasks: [
	{
		任务一
	}, 
	{
		任务二
	}
]
```

下面是每个task用到的属性

* label：label 属性是任务的名称或标识符。它用于唯一标识任务并提供可读性。使用它在配置文件中引用或区分任务

* type

  * type 属性指定了任务的类型，它告诉VS Code 如何执行任务
  * 常见的类型包括 "shell"（使用Shell执行命令）、"process"（直接运行可执行文件）、"npm"（用于运行npm脚本）等。对于C++程序cppbuild等于shell的别名

* command

  * command 属性指定（终端中）要执行的命令或操作。这是任务的主要目标
  * 通常需要提供一个可执行文件的路径或一个命令行命令

* windows：windows 属性用于指定仅在 Windows 操作系统上运行的任务。可以在  windows 下定义 Windows 特定的任务配置

* group：group 属性用于将任务分组。它包括两个子属性：kind 和 isDefault

  * kind 指定了任务的种类，如 "build"、"test"、"clean" 等
  * isDefault 用于标记是否是默认任务，当你在VS Code中运行任务时，默认任务将首先执行

* presentation

  * presentation 属性用于控制任务在 VS Code 中的呈现方式
  * 可以包括 `echo` 属性，用于控制任务的输出是否显示在输出面板中

* options

  * options 属性包含一组任务执行选项，通常用于配置任务的行为

  * 可以设置环境变量、工作目录、环境路径等选项

    * cwd, current working directory：cwd 属性用于指定任务在哪个目录下执行。这对于需要在特定目录中运行任务的情况非常有用。比方说如果任务需要在项目的根目录下执行，则可以将 `cwd` 设置为项目的根目录路径 `${workspaceFolder}`

    * env 环境变量：env 属性用于设置任务执行时的环境变量。可以指定键值对，以配置任务执行环境中的特定变量。这在需要自定义环境变量的任务中非常有用

    * shell（Shell 配置）：shell 属性允许配置任务的 Shell 环境。可以指定 Shell 的路径或使用系统默认的 Shell

      ```json
      "options": {
          "shell": {
              "executable": "/bin/bash",
              "args": ["-c"]
          }
      }
      ```

    * suppressTaskName（禁止任务名称）

      * `suppressTaskName` 属性用于控制是否在输出面板中显示任务名称
      * 如果设置为 `true`，任务的名称将不会显示在输出中，只会显示命令的输出

* runOptions

  * runOptions 属性是任务运行时的选项，通常用于配置任务在后台运行或前台运行等
  * 一些任务可能需要以不同的方式运行，runOptions 允许我们进行定制

* problemMatcher：定义如何匹配和处理任务输出中的问题（错误和警告）的设置。它允许你配置 VS Code 如何识别和展示来自任务的错误和警告信息，以便在开发过程中更容易地发现和解决问题

* detail：任务的详细说明

### 预定义变量

vscode 中的配置是可以移植的，若想让配置更加通用，最好使用一些VS Code预定义的一些变量。它们可以在配置文件（如 `settings.json`、`tasks.json` 等）中使用

* `${workspaceFolder}`： 表示当前打开的工作区（项目）的根目录路径。这个变量经常用于配置文件中，用于引用项目文件的路径或设置任务的工作目录
* `${workspaceFolderBasename}` 表示当前打开的工作区的基本名称，但是不带 `/`
* `${file}` 表示当前打开的文件的完整路径
* `${fileWorkspaceFolder}` 表示包含当前打开文件的工作区的根目录路径。这个变量可以用于确定当前文件所属的工作区，适用于多工作区项目
* `${relativeFile}` 表示当前打开文件相对于工作区根目录的路径
* `${relativeFileDirname}` 表示当前打开文件所在目录相对于工作区根目录的路径
* `${fileBasename}` 表示当前打开文件的基本名称（不包括路径和扩展名）
* `${fileBasenameNoExtension}` 表示当前打开文件的基本名称，但不包括扩展名
* `${fileDirname}` 表示当前打开文件所在的目录的路径。
* `${fileExtname}` 表示当前打开文件的扩展名（包括点号）
* `${cwd}` 表示当前工作目录的路径
* `${lineNumber}` 表示当前光标所在位置的行号
* `${selectedText}` 表示当前选中的文本。这可用于将选中的文本传递给外部工具或任务
* `${execPath}` 表示执行VS Code的可执行文件的路径
* `${defaultBuildTask}` 表示默认的构建任务的名称。如果在 `tasks.json` 中指定了默认任务，它将包含该任务的名称
* `${pathSeparator}` 表示文件路径中使用的路径分隔符，根据操作系统的不同而变化（例如，Windows 上为`\`，Linux/OS X 上为`/`）。这可用于在配置文件中指定跨平台兼容的路径

### `tasks.json` 示例

1. **打开命令面板**：使用 `Ctrl+Shift+P`（在Mac上是 `Cmd+Shift+P`）。
2. **运行任务配置**：输入“Tasks: Configure Task”并选择它。
3. **选择模板**：选择一个合适的模板，如`C/C++: g++ build active file`，来编译C++文件。
4. **调整任务**：根据你的需要调整生成的`tasks.json`文件。例如，你可能需要更改编译器标志或添加额外的命令。

```json
{
    "tasks": [
        {
            "type": "cppbuild",
            "label": "C/C++: gcc build active file",
            "command": "/usr/bin/gcc",
            "args": [
                "-fdiagnostics-color=always",
                "-g",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}"
            ],
            "options": {
                "cwd": "${fileDirname}"
            },
            "problemMatcher": [
                "$gcc"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "detail": "Task generated by Debugger."
        },
        {
            "type": "cmake",
            "label": "CMake: configure",
            "command": "configure",
            "problemMatcher": [],
            "detail": "CMake template configure task"
        }
    ],
    "version": "2.0.0"
}
```

## *run & debug*

### `launch.json` 配置调试

控制如何启动调试会话

1. **打开调试视图**：点击左侧工具栏上的“运行和调试”图标。
2. **创建`launch.json`文件**：点击“创建 launch.json 文件”链接，或点击命令面板中的“Debug: Open launch.json”。
3. **选择环境**：如果出现提示，请选择一个环境，如“C++ (GDB/LLDB)”。
4. **调整配置**：根据你的项目调整生成的`launch.json`文件。例如，你可能需要设置`program`属性来指定要调试的可执行文件的路径。

### `launch.json` 示例

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Launch Program",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/myProgram",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "build my project",
            "miDebuggerPath": "/usr/bin/gdb"
        }
    ]
}
```

## *setting.json*

### setting.json文件结构

VS Code 中的设置结构是以 JSON 格式组织的，用于配置编辑器的各种选项和行为。这些设置可以分为全局设置、工作区设置和项目设置，以满足不同层次的配置需求。

1. **全局设置 (User Settings)**：全局设置是应用于整个 VS Code 编辑器的设置，对所有工作区和项目都生效。全局设置存储在全局 `settings.json` 文件中，通常位于用户配置文件夹中，具体位置根据操作系统不同而有所不同。可以在 VS Code 中使用 "文件" -> "首选项" -> "设置"（或快捷键 `Ctrl` + `,`）打开全局设置
2. **工作区设置 (Workspace Settings)**：工作区设置是针对特定工作区（项目）的设置。在Ubuntu上这些设置存储在工作区根目录下的 `/home/$USER/.vscode-server/data/Machine/settings.json` 文件中。工作区设置仅适用于当前工作区，可以覆盖全局设置。工作区设置通常包括项目特定的配置，例如构建任务、调试配置等
3. **项目设置 (Folder/Project Settings)**：项目设置是在工作区内的特定文件夹或子目录下的设置，存储在 `.vscode/settings.json` 文件中。这允许为项目的不同部分或子项目设置不同的配置

这些设置以 JSON 格式组织，每个设置都有一个键（key）和一个对应的值（value）。可以在这些设置中配置编辑器的各种选项、扩展的行为、编辑器风格等。VS Code 提供了强大的用户界面来编辑这些设置，也可以手动编辑配置文件

注意：在项目级别的设置文件中，只需包含需要自定义的设置，而不需要将所有设置都复制到项目文件中。只有在设置文件中指定的设置项才会覆盖全局设置

这种组织结构允许你根据不同的层次和需求来配置 VS Code，使其适应不同的工作流和项目

### c_cpp_properties.json

VS Code为C++项目提供了强大的支持

`c_cpp_properties.json` 是用于配置 C/C++ 项目的 VS Code 设置文件之一。它主要用于指定项目的编译器路径、包含目录、宏定义以及其他与代码分析和 IntelliSense 相关的配置信息。这个文件通常用于确保 VS Code 正确识别和分析你的 C/C++ 代码，以提供代码补全、代码导航和错误检查等功能。

以下是 `c_cpp_properties.json` 文件的一些常见配置项和示例：

```json
{
  "configurations": [
    {
      "name": "Linux",
      "includePath": [
        "${workspaceFolder}/**",
        "/usr/include",
        "/usr/local/include"
      ],
      "defines": [
        "_DEBUG",
        "UNICODE",
        "__GNUC__"
      ],
      "compilerPath": "/usr/bin/gcc",
      "cStandard": "c11",
      "cppStandard": "c++17",
      "intelliSenseMode": "gcc-x64",
      "browse": {
        "path": [
          "${workspaceFolder}",
          "/usr/include",
          "/usr/local/include"
        ],
        "limitSymbolsToIncludedHeaders": true,
        "databaseFilename": ""
      }
    }
  ],
  "version": 4
}
```

* `"configurations"`：这是一个数组，包含了不同编译配置的设置。每个配置都有一个 `"name"` 字段，用于标识配置的名称。
* `"includePath"`：这是一个数组，包含了要用于代码分析的包含目录路径。通常包括项目目录 `"${workspaceFolder}"` 和标准库的包含目录。
* `"defines"`：这是一个数组，包含了预定义的宏定义。它们用于配置代码分析和 IntelliSense 的行为。
* `"compilerPath"`：指定了编译器的路径。这是为了确保 VS Code 使用正确的编译器进行代码分析和 IntelliSense。
* `"cStandard"` 和 `"cppStandard"`：指定了 C 和 C++ 的标准版本，用于代码分析和 IntelliSense 的配置。
* `"intelliSenseMode"`：指定 IntelliSense 使用的模式，例如 `"gcc-x64"` 表示使用 GCC 编译器。
* `"browse"`：这个部分用于配置符号浏览器的设置，包括浏览路径、符号数据库文件等。

请注意，`c_cpp_properties.json` 文件通常是针对每个项目或工作区创建的，因此你可能会在不同的项目中看到不同的配置。你可以在 VS Code 中使用工作区设置或项目设置来配置 `c_cpp_properties.json` 文件，以适应不同的编译环境和需求。这有助于确保代码分析和 IntelliSense 在不同的项目中都能正常工作。

# 远程调试



