# 安装 & 编译 & 测试GCC

[GCC, the GNU Compiler Collection - GNU Project](https://gcc.gnu.org/)

[Installing GCC - GNU Project](https://gcc.gnu.org/install/)

GCC没有采用cmake编译，而是用了autoconfi-make的编译方式，所以和LLVM相比GCC自身的编译和安装没有那么简单，需要注意很多细节。我们用 `srcdir` 来指代GCC的source directory（`MAINTAINERS` 文件所在的目录），用 `objdir` 来指代toplevel `build/object` 目录

## *Prerequisites*

### 拉取GCC

[gcc-mirror/gcc at releases/gcc-14.1.0 (github.com)](https://github.com/gcc-mirror/gcc/tree/releases/gcc-14.1.0)

截止到2024.07.02，最新的stable edition为GCC 14.1

[GCC Development Plan - GNU Project](https://gcc.gnu.org/develop.html#timeline)

```cmd
$ git clone --branch releases/gcc-14.1.0 --depth 1 https://github.com/gcc-mirror/gcc.git gcc14.1.0
$ git clone --branch releases/gcc-4.9.0 --depth 1 https://github.com/gcc-mirror/gcc.git gcc4.9.0
```

或者慢的话可以用阿里的镜像：[gnu-gcc安装包下载_开源镜像站-阿里云 (aliyun.com)](https://mirrors.aliyun.com/gnu/gcc/)

```cmd
$ wget https://mirrors.aliyun.com/gnu/gcc/
```

### 安装依赖

GMP (GNU Multiple Precision Arithmetic Library), MPFR (multiple-precision floating-point computations) and MPC (Complex numbers) 

```cmd
$  ./contrib/download_prerequisites
```

在srcdir运行上面的script，如果中间报了下面的错误，就是由于网络连接问题导致文件没有拉全，多拉几次就行

```
WARNING: 1 computed checksum did NOT match
error: Cannot verify integrity of possibly corrupted file
```

## *Procedure for native compiler*

### Native Compiler

Native compiler 是和 cross compiler 相对w的概念，它指的是一个编译器，它产出的代码是为运行该编译器的本地系统架构而优化的。这意味着，该编译器生成的可执行文件是直接针对当前计算机的 CPU 架构（比如 x86, ARM, MIPS 等）设计的，并可以在没有任何模拟或虚拟化层的情况下执行

比如说如果在一个 x86_64 架构的 Linux 系统上使用 GCC 编译器编译 C 代码，那么这个 GCC 就可以被认为是一个 native compiler，因为它默认会生成专门为 x86_64 架构优化的二进制程序

### 配置 & 编译

```cmd
$ mkdir objdir
$ cd objdir
$ ../configure --enable-languages=c,c++,fortran --prefix=/usr/local/gcc-14.1.0 --disable-multilib
$ make -j15    # Check the number of CPUs by running grep -w processor /proc/cpuinfo|wc -l. In this example, the number is 15. You can set the parameters as required.0
$ make install 
```

GCC采用自举编译

注意我们用的是 `srcdir/configure`，而不是直接 `./configure`，因为

[InstallingGCC - GCC Wiki (gnu.org)](https://gcc.gnu.org/wiki/InstallingGCC)

> A major benefit of running *srcdir*`/configure` from outside the source directory (instead of running `./configure`) is that the source directory will not be modified in any way, so if your build fails or you want to re-configure and build again, you simply delete everything in the *objdir* and start again.

### Test (optional)

## *安装多个版本的GCC*

若系统上安装了多个版本的 GCC，并且想要设置默认使用的版本的话有下面几种方案

### 更新 PATH 环境变量

修改 `PATH` 环境变量，将希望作为默认 GCC 版本的 `bin` 目录放在 `PATH` 的前面。例如，如果想将 `/usr/local/gcc-9.2/bin` 设为默认：

```cmd
$ export PATH=/usr/local/gcc-9.2/bin:$PATH
```

这一行可以加入到 shell 配置文件中（比如 `~/.bashrc`、`~/.bash_profile` 或 `~/.profile` 等），然后重新登录或者运行 `source ~/.bashrc` 来应用更改

### 使用 update-alternatives（Debian/Ubuntu 系统）

Debian 和 Ubuntu 使用 `update-alternatives` 系统来管理同一命令的多个版本，它是专门维护系统命令链接符的工具。要使用这个方法，首先要为每个 GCC 版本设置一个替代选项：

* install：增加一组新的系统命令链接符

  ```cmd
  $ update-alternatives --install <link> <name> <path> <priority> [--slave link name path]
  ```

  - `<link>`：这指的是由 `update-alternatives` 创建和管理的符号链接。通常这会是一个通用命令名（如 `/usr/bin/editor`），当用户运行该命令时，符号链接会指向用户选择的特定程序版本
  - `<name>`：这是替代群组的名称。所有提供类似功能的可替换项都会被分配同一个名称
  - `<path>`：这是到真实的二进制执行文件或脚本的绝对路径。当这个替代项被选择为默认时，`<link>` 指向这个 `<path>`
  - `<priority>`：这是一个整数值，用于在自动模式下决定哪个替代项成为默认项。具有最高优先级的替代项将被设置为默认项。
  - `--slave <link> <name> <path>`：这是一个可选参数，允许你同时设置“从属”的链接。当主链接变化时，“从属”链接也会相应改变。例如，当你更新默认的编辑器时，你可能也想更新默认的手册页编辑器。

  ```cmd
  $ sudo update-alternatives --install /usr/local/bin/gcc gcc /usr/bin/gcc 50
  $ sudo update-alternatives --install /usr/local/bin/gcc gcc /usr/local/gcc-14.1.0/bin/gcc 60
  $ sudo update-alternatives --install /usr/local/bin/g++ g++ /usr/bin/g++ 50
  $ sudo update-alternatives --install /usr/local/bin/g++ g++ /usr/local/gcc-14.1.0/bin/g++ 60
  ```

* display：display选项用来显示一个命令链接的所有可选命令，即查看一个命令链接组的所有信息，包括链接的模式（自动还是手动）、链接priority值、所有可用的链接命令等等

* remove：删除一个命令的link值，其附带的slave也将一起删除

* config：显示和修改实际指向的候选命令，为在现有的命令链接选择一个作为系统默认

### 修改符号链接

如果你不想使用 `update-alternatives`，也可以直接创建或修改 `/usr/bin` 中的符号链接指向你选择的 GCC 版本。请记住，在对 `/usr/bin` 下的文件进行操作前应非常小心，因为这可能会影响系统中其他程序的正常运行。例如：

```cmd
$ sudo ln -sf /usr/local/gcc-9.2/bin/gcc /usr/bin/gcc
$ sudo ln -sf /usr/local/gcc-9.2/bin/g++ /usr/bin/g++
```

### 模块管理工具

在某些系统中，尤其是HPC环境中，可能会使用模块环境管理工具（如 Environment Modules 或 Lmod）。这些工具允许用户动态修改环境变量，例如 `PATH`、`LD_LIBRARY_PATH` 等。如果系统中有模块化环境，可以加载相应的模块来设置默认的 GCC 版本

```cmd
$ module load gcc-9.2
```

## *配置选项*

[Installing GCC: Configuration - GNU Project](https://gcc.gnu.org/install/configure.html)

支持的配置选项可以在srcdir用 `./configure --help` 来查看

### 安装目录

* `--prefix=PREFIX`：安装 architecture-independent的文件到PREFIX目录下，默认是 `/usr/local`。一般会设置为 `/usr/local/gcc_edition`，因为很可能会在系统上安装多个版本的gcc
* `--exec-prefix=EPREFIX`：安装 architecture-dependent的文件到EPREFIX目录下
* `--bindir=DIR`：设置用户可执行文件的安装目录，默认通常是 `EPREFIX/bin`。这是普通用户将使用的二进制程序存放的地方
* `--sbindir=DIR`：设置系统管理员使用的可执行文件的安装目录，默认通常是 `EPREFIX/sbin`。这里包含了系统管理相关的程序
* `--libexecdir=DIR`：设置应用程序可执行文件的安装目录，默认通常是 `EPREFIX/libexec`。这些不直接被用户调用的可执行文件一般由其他程序内部使用
* `--sysconfdir=DIR`：设置只读的单机数据文件（如配置文件）的安装目录，默认通常是 `PREFIX/etc`
* `--sharedstatedir=DIR`：设置可修改的、与架构无关的数据文件的安装目录，默认通常是 `PREFIX/com`
* `--localstatedir=DIR`：设置可修改的、只针对单个机器的数据文件的安装目录，默认通常是 `PREFIX/var`。这包括像日志文件这样的变化数据
* `--libdir=DIR`：设置对象代码库文件（如 `.so`, `.a` 文件）的安装目录，默认通常是 `EPREFIX/lib`
* `--includedir=DIR`：设置 C 头文件的安装目录，默认通常是 `PREFIX/include`。开发者使用的头文件通常放在这里
* `--oldincludedir=DIR`：设置非 GCC 编译器使用的 C 头文件的安装目录，默认通常是 `/usr/include`
* `--datarootdir=DIR`：设置只读的、与架构无关的数据文件根目录，默认通常是 `PREFIX/share`
* `--datadir=DIR`：设置只读的、与架构无关的数据文件的安装目录，默认基于 `DATAROOTDIR`
* `--infodir=DIR`：设置 info 文档的安装目录，默认基于 `DATAROOTDIR/info`
* `--localedir=DIR`：设置依赖于地区设置的数据（如本地化信息）的安装目录，默认基于 `DATAROOTDIR/locale`
* `--mandir=DIR`：设置 man 手册页的安装目录，默认基于 `DATAROOTDIR/man`
* `--docdir=DIR`：设置文档根目录，默认基于 `DATAROOTDIR/doc/PACKAGE`，其中 `PACKAGE` 通常是软件包的名称
* `--htmldir=DIR`：设置 html 格式文档的安装目录，默认基于 `DOCDIR`
* `--dvidir=DIR`：设置 dvi 格式文档的安装目录，默认基于 `DOCDIR`
* `--pdfdir=DIR`：设置 pdf 格式文档的安装目录，默认基于 `DOCDIR`
* `--psdir=DIR`：设置 ps（PostScript）格式文档的安装目录，默认基于 `DOCDIR`

### 一些开关

* `--enable-languages=lang1,lang2,...`：只需要编译某些语言的前端和runtime

  可以在 `srcdir/gcc` 下用 `grep ^language= */config-lang.in` 来查看所有gcc支持的语言



* `--enable-multilib` & `--disable-multilib`

电脑默认的编译配置是32位和64位，但是32位的dev lib不齐全，建议最好关掉32位，进行如下操作

```
configure: error: I suspect your system does not have 32-bit development libraries (libc and headers). If you have them, rerun configure with --enable-multilib. If you do not have them, and want to build a 64-bit-only compiler, rerun configure with --disable-multilib.
```



## *Cross-compile*

### configure

  --build=BUILD     configure for building on BUILD [guessed]

* `--host=HOST`       cross-compile to build programs to run on HOST [BUILD]
* `--target=TARGET`   configure for building compilers for TARGET [HOST]

# 编译选项

### 链接选项

[gcc 编译参数 -fPIC 的详解和一些问题_gcc fpic参数-CSDN博客](https://blog.csdn.net/a_ran/article/details/41943749)

* -fPIC & -fpic都是在编译时加入的选项，用于生成位置无关代码 Position-Independent-Code。这两个选项都是可以使代码在加载到内存时使用相对地址，所有对固定地址的访问都通过GOT来实现。-fPIC & -fpic最大的区别在于是否对GOT的大小有限制。-fPIC对GOT表大小无限制，所以如果在不确定的情况下，使用-fPIC是更好的选择，它可以生成更高效的代码
* -fPIE与-fpie是等价的。这个选项与-fPIC/-fpic大致相同，不同点在于：-fPIC用于生成动态库，-fPIE用与生成可执行文件。再说得直白一点：-fPIE用来生成位置无关的可执行代码

### 路径搜索

最常用的就是 `-I, -isystem, -iquote, -idirafter`

用 `-iquote` 指定的目录仅适用于引号形式的指令，即 `#include "file"`。用 -I、-isystem 或 -idirafter 指定的目录适用于 `#include "file"` 和 `#include <file>` 两种形式的指令查找

可以在命令行上指定任意数量或组合这些选项来搜索多个目录中的头文件。搜索顺序如下：

1. 对于引号形式的 #include，首先搜索当前文件的目录
2. 对于引号形式的 #include，按照命令行上出现的从左到右的顺序，搜索由 `-iquote` 选项指定的目录
3. 按照从左到右的顺序扫描用 `-I` 选项指定的目录
4. 按照从左到右的顺序扫描用 `-isystem` 选项指定的目录
5. 扫描标准系统目录
6. 按照从左到右的顺序扫描用 `-idirafter` 选项指定的目录

可以使用 `-I` 来覆盖系统头文件，替换为我们自己的版本，因为这些目录在标准系统头文件目录之前被搜索。**不应该使用这个选项来添加包含 vendor-supplied 系统头文件的目录，对此应该使用 `-isystem`**

-isystem 和 -idirafter 选项还会将目录标记为系统目录，使其获得与标准系统目录相同的特殊处理。

如果一个标准系统包含目录，或者用 -isystem 指定的目录也用 -I 指定了，那么 -I 选项将被忽略。该目录仍会以其在系统包含链中的正常位置作为系统目录进行搜索。这是为了确保修复有问题的系统头文件的 GCC 程序以及 #include_next 指令的排序不会意外改变。如果你真的需要更改系统目录的搜索顺序，请使用 -nostdinc 和/或 -isystem 选项。

-I- 分割包含路径。这个选项已经被弃用。请改用 -iquote 替代 -I- 之前的 -I 目录，并移除 -I- 选项。

在 -I- 之前用 -I 选项指定的任何目录只会为 #include "file" 请求的头文件进行搜索；它们不会为 #include <file> 进行搜索。如果在 -I- 之后用 -I 选项指定了额外的目录，那么那些目录会为所有的 ‘#include’ 指令进行搜索。

此外，-I- 还禁止使用当前文件目录作为 #include "file" 的第一个搜索目录。没有办法覆盖 -I- 的这种效果。



### Enable Warnings

`-W*` 开头的编译选项用于开启编译中的Warnings

`-Wall` 并不会开启所有可能的警告，而是开启了以下一组被认为最有用的警告：

- `-Waddress`: 警告如果一个表达式总是真的或假的（比如数组永远不会被当作false）。
- `-Warray-bounds` (only with `-O2`): 警告数组下标越界。
- `-Wc++11-compat` 和 `-Wc++14-compat`: 警告C++代码与C++11和C++14标准不兼容的地方。
- `-Wchar-subscripts`: 警告如果一个字符类型被用作数组下标。
- `-Wenum-compare`: 在C++程序中警告如果枚举类型之间进行比较。
- `-Wimplicit-int`: 在C语言中警告如果声明函数时没有指定返回类型，默认会当作int处理。
- `-Wimplicit-function-declaration`: 警告函数在使用前未被声明。
- `-Wcomment`: 警告嵌套的块注释 `/* ... /* ... */`。
- `-Wformat` 和 `-Wformat-security`: 警告格式字符串不匹配相应参数类型的情况，以及可能的安全问题。
- `-Wmissing-braces`: 警告在数组初始化时括号可能遗漏的地方。
- `-Wnonnull`: 警告传递给需要非空参数的函数的参数是空的。
- `-Wparentheses`: 警告可能因优先级不明确造成歧义的地方。
- `-Wpointer-sign`: 警告指针类型之间赋值时的符号不匹配。
- `-Wreorder`: 在C++程序中，警告成员初始化列表的顺序与成员声明的顺序不一致。
- `-Wreturn-type`: 警告函数没有返回语句或返回了错误类型的值。
- `-Wsequence-point`: 警告顺序点相关的问题，比如多次改变一个变量的值而不通过顺序点。
- `-Wsign-compare`: 警告符号比较中可能出现的问题，例如unsigned和signed值之间的比较。
- `-Wstrict-aliasing`: 针对可能由于别名规则导致的问题的警告。
- `-Wswitch`: 警告在switch语句中枚举值没有相对应的case。
- `-Wtrigraphs`: 警告三字符序列，这些是遗留特性，可能会引起混淆。
- `-Wunused`: 警告任何未使用的变量。
- `-Wuninitialized`: 使用 `-O1` 或以上优化等级时，警告未初始化的变量。
- `-Wunknown-pragmas`: 警告不被识别的预处理指令 #pragma。

# 架构



<img src="GCC-Architecture.png" width="70%">



### gcc vs. g++

gcc和g++的主要区别默认的编程语言和链接库

虽然gcc和g++都可以编译C和C++，但是gcc默认编译C，而g++则默认编译C++。g++会自动链接C++标准库，而gcc则不会

gcc和g++都有很多预定义的宏，但是数目和内容则不同。比如下面这些gcc的预定义宏

```C
#define __GXX_WEAK__ 1
#define __cplusplus 1
#define __DEPRECATED 1
```





-Wall参数，可以开启警告信息，显示所有的警告信息

# IR

## *GIMPLE*

GIMPLE Generic and GIMPLE Intermediate Language 是GCC使用的三地址码

## *RTL*

RTL, Register Transfer Language

# Optimization

### GCC 优化等级

<img src="gcc优化等级.png" width="50%">

1. **-O0**：没有优化。此等级生成最简单、最容易调试的代码，但性能通常较低。
2. **-O1**：启用基本的优化。这一级别启用一些简单的优化，如函数内联和一些代码移动。它可以提高性能，同时保留了较好的调试能力。
3. **-O2**：启用更多的优化。此等级会应用更多的代码转换，包括循环展开和更强大的优化。这通常会提高代码的性能，但会增加编译时间。
4. **-O3**：启用高级优化。它启用了大多数常用的优化，包括函数内联、循环展开、自动矢量化等。这可以显著提高生成的代码的性能，但仍然保留了对浮点精度和符号运算的一定程度的保守性。因此，它适用于大多数情况下，可以在不牺牲太多数值精度的情况下提高性能。
5. **-Ofast**：极高级别的优化。此等级启用了所有常见的优化，同时允许牺牲一些数值精度以提高性能。它适用于那些对数值精度要求不高的高性能应用。
6. **-Os**：优化代码大小。这一级别旨在减小生成的可执行文件的大小，而不是提高性能。它会删除一些不必要的代码和数据，适合于资源有限的环境。
7. **-Og**：适用于调试的优化。这一级别会进行一些优化，同时保留了较好的调试能力。它是为了在调试期间获得较好的性能和调试能力的平衡。
8. **-O**：默认优化等级。这一级别通常等同于 `-O1` 或 `-O2`，具体取决于编译器版本和配置。

一些参数的意义如下

* `-flto`

  在GCC编译器中，选项 `-flto` 表示 "Link Time Optimization"，即链接时优化。它是一种编译器优化技术，它将编译阶段的优化延伸到链接阶段，以进一步提高生成的可执行文件的性能。

  使用 `-flto` 选项，编译器将在编译时生成中间表示（IR），然后将这些中间表示保存在目标文件中。在链接时，编译器会再次优化这些中间表示，并生成最终的可执行文件。这使得编译器能够进行全局的优化，跨足够多的源文件，从而产生更高效的代码。

* `march=native`

  `march=native` 是GCC编译器的一个选项，用于优化生成的机器代码以最大限度地利用当前主机的CPU架构。这个选项告诉编译器使用当前主机的本机（native）CPU架构，以便生成特定于该CPU的指令集的代码。

  使用 `-march=native` 时，GCC会检测当前主机的CPU架构，并根据检测结果生成与该架构最兼容的机器代码。这可以提高程序的性能，因为生成的代码会更好地利用当前CPU的特性和指令集扩展。

  注意：使用 `-march=native` 选项可能会导致生成的代码在其他CPU架构上不兼容，因为它会针对当前主机的CPU进行优化。因此如果计划在多个不同CPU架构的计算机上运行相同的二进制程序，应谨慎使用这个选项。

  这个选项通常用于在特定主机上编译和运行程序，以获得最佳性能。**如果要生成可移植的代码，不建议使用 `-march=native`**，而应选择适当的目标架构标志，例如 **`-march=core2`**、**`-march=corei7`** 等。这将生成适用于特定CPU架构的代码，而不仅仅是当前主机的本机架构。





# 预定义宏

## *`__attribute__`*

### GCC 的 \_\_attribute\_\_ 属性说明符

[Function Attributes - Using the GNU Compiler Collection (GCC)](https://gcc.gnu.org/onlinedocs/gcc-5.3.0/gcc/Function-Attributes.html#Function-Attributes)

`__attribute__` 机制是 GNU C 编译器（如 GCC）提供的一种用于控制编译器行为和注释的机制。它允许程序员使用一些特殊的属性来**告诉编译器**如何处理变量、函数、结构等元素，或者对代码进行一些特殊的优化或警告。比如它们的对齐方式、是否进行内联展开、是否在链接时可见等

`__attribute__` 的格式为

```C
__attribute__ ((attribute-list))
```

 `__attribute__`可以设置函数属性 Function Attribute 、变量属性 Variable Attribute 和类型属性 Type Attribute

### `__attribute__` 的实现

`__attribute__` 机制是通过宏的多层封装实现的

### 函数属性

* `__attribute__((noreturn))`: 这个属性用于标记函数，表示该函数不会返回。这对于像 `exit()` 这样的函数很有用，因为它们在调用之后程序将终止，从不会返回

  ```c
  void my_exit() __attribute__((noreturn));
  ```

* `__attribute__((constructor))` 和 `__attribute__((destructor))`: 这些属性用于标记函数，指示它们应该在程序启动或结束时自动执行，通常用于初始化或清理工作

  ```c
  void my_init_function() __attribute__((constructor));
  void my_cleanup_function() __attribute__((destructor));
  ```

* `__attribute__((warn_unused_result))`: 这个属性用于标记函数，表示调用该函数的返回值应该被检查，以避免警告。

  ```c
  int get_value() __attribute__((warn_unused_result));
  ```

* `__attribute__((clean_up))` [黑魔法__attribute__((cleanup)) · sunnyxx的技术博客](https://blog.sunnyxx.com/2014/09/15/objc-attribute-cleanup/)

### 变量属性

### 自定义段

正常情况下，GCC编译出来的目标文件中，代码文本会被放到 `.text` 段，全局变量和静态变量则会被到 `.data` 和 `.bss` 段。但是有时候可能希望变量或某些部分代码能够放到自己所指定的段中去，以实现某些特定的功能。比如为了满足某些硬件的内存和IO地址布局，或者是像Linux操作系统内核中用来完成一些初始化和用户空间复制时出现页错误异常等

GCC提供了一个扩展机制，使得程序员可以指定变量所处的段

```c++
__attribute__((section("FOO"))) int global = 42;
__attribute__((section("BAR"))) void foo() {}
```

### 类型属性

* `__attribute__((packed))`: 这个属性用于结构体，它告诉编译器要尽量减小结构体的内存占用，不要进行字节对齐

  ```c
  struct MyStruct {
      int a;
      char b;
  } __attribute__((packed));
  ```

* `__attribute__((unused))`: 这个属性可以用于变量或函数，它告诉编译器忽略未使用的警告

  ```c
  int unused_variable __attribute__((unused));
  ```

* `__attribute__((aligned(N)))`: 这个属性用于指定变量或结构体的对齐方式，其中 `N` 是对齐要求的字节数

  ```c
  int aligned_variable __attribute__((aligned(16)));
  ```

### 同时使用多个属性

可以在同一个函数声明里使用多个 `__attribute__`，并且实际应用中这种情况是十分常见的。使用方式上，可以选择两个单独的 `__attribute__`，或者把它们写在一起

```C++
/* 把类似printf的消息传递给stderr 并退出 */
extern void die(const char *format, ...)
   __attribute__((noreturn))
   __attribute__((format(printf, 1, 2)));

// 或者写成

extern void die(const char *format, ...)
   __attribute__((noreturn, format(printf, 1, 2)));
```

如果带有该属性的自定义函数追加到库的头文件里，那么所以调用该函数的程序都要做相应的检查

## *alias机制*

https://www.cnblogs.com/justinyo/archive/2013/03/12/2956438.html

在 *C及其链接装载.md* 中介绍过了强弱符号的概念，GCC提供了alias机制来让用户干预

定义在 `include/libc-symbols.h` 下面

`__attribute__((alias))` 用于创建一个符号别名，将一个变量、函数或符号关联到另一个符号上。这可以用于在编译期间将一个符号的名称关联到另一个名称，从而使它们在链接时被视为同一符号

```c
/* Define ALIASNAME as a strong alias for NAME.  */
# define strong_alias(name, aliasname) _strong_alias(name, aliasname)
# define _strong_alias(name, aliasname) \
  extern __typeof (name) aliasname __attribute__ ((alias (#name))) \
    __attribute_copy__ (name);

/* Define ALIASNAME as a weak alias for NAME.
   If weak aliases are not available, this defines a strong alias.  */
# define weak_alias(name, aliasname) _weak_alias (name, aliasname)
# define _weak_alias(name, aliasname) \
  extern __typeof (name) aliasname __attribute__ ((weak, alias (#name))) \
    __attribute_copy__ (name);
```

### 强别名 strong_alias

在大多数情况下，当在C或C++代码中定义一个函数或变量时，它默认具有强链接属性。这意味着如果多个不同的编译单元（比如不同的源文件）尝试定义同名的全局符号，链接器会报告错误，因为它不允许有多个相同名称的强符号存在

* 使用 `__attribute__((alias))` 属性创建强别名
* 强别名会将一个符号完全替代为另一个符号，它们在链接时被视为完全相同的符号，没有区别。强别名会完全替代原始符号，因此它们具有相同的可见性和强度
* 如果两个符号具有相同的名称，则 `strong_alias` 可以用于将它们显式地关联在一起

### 弱别名 weak_alias

与强符号不同，一个弱符号允许在程序中存在多个同名的定义。只要至少有一个是强定义，链接器就不会报错，而是选择强符号的定义来解析所有引用。如果所有的符号都是弱的，则链接器会从它们中选择任意一个

* 使用 `__attribute__((weak))` 属性创建弱别名
* 弱别名不会完全替代原始符号，而是在原始符号不存在时才会起作用
* 如果原始符号存在，弱别名将被忽略，原始符号将被使用。弱符号在库设计中非常有用，特别是当库希望提供一些可选的、可由用户覆盖的默认行为时。此外，弱符号也用于实现某些运行时功能，如动态链接的函数替换等
* 弱别名常用于提供一个**默认实现**，但允许用户覆盖它

```C
int myFunction() {
    // 实现
}

int __attribute__((weak)) myFunction() {
    // 默认实现
}
/*========================================*/
int original_myFunction() {
    // 实际实现
}

// 创建一个弱别名
extern int alias_myFunction() __attribute__((weak, alias("original_myFunction")));
```

###  弱引用 `__attribute__((weakref))`

## *Visibility*

`visibility`属性是GCC和Clang都支持的一个编译器属性，用于设置符号在共享库中的可见性级别。这个属性影响了如何处理函数、变量及类型定义对于动态链接器和运行时的可见性。主要目的是减少动态库的大小和提高动态加载的性能

### Visibility Levels

以下是一些常见的可见性选项：

- `default`: 这是系统默认的可见性。除非另外指定，所有对象都被赋予此可见性。它表示符号将被动态链接器看到，可以从其他模块（例如其他共享库或执行文件）被引用
- `hidden`: 此选项会隐藏符号，使其无法被其他模块直接引用。即使头文件被公开包含，并且相应的API被公开调用，如果该符号被标记为`hidden`，在其他模块中将不可见。这有助于避免名称空间的冲突，并可以减少动态符号的数量，降低动态链接的成本
- `protected`: 符号以受保护的方式被导出。这意味着符号对于外部模块不可见，但是对于定义它的模块内部的其他符号则是可见的。这种情况下的符号解析速度比默认可见性更快，因为它们不需要经过动态链接器来解析
- `internal`: 类似于`hidden`，但它仅适用于ELF格式的文件。这会使得符号不能被外部模块引用，同时也不能在所定义的模块内通过函数指针进行引用

### Example

```C
__attribute__((visibility("default"))) void myFunction() {
    // 函数实现
}

__attribute__((visibility("hidden"))) int hiddenVariable;

typedef struct __attribute__((visibility("hidden"))) {
    int internalField;
} HiddenStruct;
```

在上面的示例中，`myFunction` 被赋予了默认的可见性，所以它在其他模块中是可见的。而`hiddenVariable` 和 `HiddenStruct` 标记为 `hidden`，所以它们在模块外部是不可见的

### 注意事项

- 当使用`visibility`属性时，务必了解代码库和其他依赖关系之间的交互关系。错误地设置可见性可能会导致链接错误或运行时错误
- 如果正在创建库，考虑使用`visibility`属性来优化库的大小和性能
- `visibility`属性的使用将影响代码的移植性，因为它是特定于编译器和平台的。确保只在支持它的环境中使用它
- 在设置可见性时，请考虑应用程序的安全性，因为限制符号的可见性也可以作为防止符号被恶意代码利用的一种手段

## *其他预定义宏*

### _GNU_SOURCE

`_GNU_SOURCE` 是一个用于预处理的宏定义，它用于启用特定于GNU编译器（如GCC）和GNU C库（glibc）的扩展功能和特性。具体来说，当在程序中包含 `_GNU_SOURCE` 宏定义时，编译器和标准库会根据 GNU 扩展启用额外的功能和特性，这些功能和特性在标准C或C++中不一定可用

`_GNU_SOURCE` 宏通常用于以下情况：

* 启用 GNU C 库的扩展功能：通过定义 `_GNU_SOURCE`，可以启用 glibc 提供的一些额外的非标准函数和特性。例如，这可以包括特定于 GNU 的线程函数、内存分配函数等
* 启用 POSIX 标准扩展：一些 POSIX 扩展功能在标准C或C++中不是默认可用的。通过定义 `_GNU_SOURCE`，可以启用这些 POSIX 扩展，使它们在程序中可用
* 启用一些 GNU 编译器扩展：一些 GNU 编译器（如 GCC）提供了特定于编译器的扩展功能。通过定义 `_GNU_SOURCE`，可以启用这些扩展功能

需要注意的是，使用 `_GNU_SOURCE` 可能会使代码在不同编译器和平台上不具有可移植性，因为这些扩展功能不一定在所有编译器和标准库中都可用。因此，除非明确需要使用这些特定于 GNU 的特性，否则最好避免在通用的跨平台代码中使用 `_GNU_SOURCE`。如果要编写可移植的代码，建议仅依赖于标准C或C++功能，而不使用非标准或特定于编译器/库的扩展



# C/C++编译选项

`-fno-rtti`：禁用运行时类型信息，即不会生成 `type_info`，如果没有使用到这个特性的话可以关闭冗余的RTT特性，I来减小编译文件大小。异常处理使用相同的信息，但它会根据需要生成它。dynamic_cast 仍然可以用于不需要运行时类型信息的转换，即转换到 `void*` 或者无歧义的基类

`fno-exceptions`：禁用异常

# libstdc++

libstdc++会被默认安装在 `/usr/local/include/c++/v1` 中





GCC 9.3.0: GLIBCXX_3.4.28, CXXABI_1.3.12

# LD

LD (Link Editor or Loader) 是GNU所使用的链接器

## *GNU gold*

GNU `gold` 链接器是对传统 GNU `ld` 链接器的一种现代替代品，最初由 Ian Lance Taylor 在 Google 工作期间开发。`gold` 的主要目标是提高链接速度，特别是针对大型的 C++ 应用程序。为了达到这一目的，`gold` 采用了许多性能优化措施，例如：

- 使用线程来并行化链接过程
- 使用更有效率的数据结构和算法
- 只支持ELF文件格式，这是 Linux 和其他 Unix-like 系统上使用最广泛的文件格式

在很多情况下，`gold` 能够比 `ld` 提供更快的链接时间，这使得它在需要快速迭代编译的开发过程中非常有用。然而，`gold` 并不支持 `ld` 所有的功能，这意味着某些特殊场景下可能还需要依赖 `ld`

# MinGW

## *Win平台C/C++编译器*

### MSVC

MSVC Microsoft Visual C++ 是微软的官方C/C++编译器和开发工具集，主要用于Windows平台上的应用程序开。MSVC通常与Visual Studio集成在一起。它是Windows上最常用的编译器之一

以下是MSVC的主要部件

* 编译器：MSVC包括Microsoft的C/C++编译器，用于将C和C++源代码编译成可执行文件。这个编译器通常是针对Windows平台的性能优化的
* 开发工具包 SDK：除了编译器，MSVC提供了一套丰富的开发工具，包括调试器、性能分析器、图形用户界面设计器、资源编辑器等，这些工具可以帮助开发者创建、调试和优化Windows应用程序
* 标准库：MSVC包括Microsoft的C/C++标准库实现，这些库提供了许多标准函数和类，以便开发者能够使用标准的C/C++函数和数据结构
* Windows API支持：MSVC深度集成了Windows API（应用程序编程接口）支持，使开发者能够轻松地访问Windows操作系统的功能和服务，创建本机Windows应用程序
* 版本和兼容性：MSVC的版本随着时间的推移而不断更新，支持新的C/C++标准和Windows平台的最新功能。开发者可以选择使用不同版本的MSVC，根据他们的需求和目标平台来进行开发

### MinGW

MinGW, Minimalist GNU for Windows 是一个开源项目（Windows的极简GNU）。它的主要目标是提供一个在Microsoft Windows操作系统上使用的GNU工具链，以便开发者可以编译和运行GNU/Linux类似的软件，而无需在Windows上使用大型的商业编译器

MinGW包括以下关键组件：

* GCC：GNU编译器集合（GCC）的Windows移植版本。这是一个强大的C/C++和其他编程语言的编译器，可以将源代码编译成可在Windows上运行的可执行文件
* Binutils：GNU二进制工具集的Windows移植版本，包括汇编器、链接器和其他二进制工具，用于处理可执行文件和库文件
* MSYS：Minimal System（MSYS）是MinGW系统的一部分，提供了一个轻量级的Unix命令行环境，以帮助在Windows上构建和运行Unix风格的工具和脚本。MSYS允许在Windows上使用bash shell、make命令等

MinGW的主要优点包括：

* 免费和开源：MinGW是免费的，并且以开源方式提供，允许开发者自由使用和分发它。
* 轻量级：MinGW的目标是提供一个轻量级的工具链，不需要庞大的开发环境，因此它非常适合需要在Windows上进行C/C++开发的开发者。
* 与标准GNU工具链兼容性：MinGW允许开发者使用标准的GNU工具和命令，从而实现与Linux等Unix-like系统的兼容性。
* 可移植性：MinGW生成的可执行文件可以在Windows上运行，而无需额外的运行时库。
