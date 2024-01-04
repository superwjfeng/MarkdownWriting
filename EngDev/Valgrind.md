https://valgrind.org

Valgrind 是一个GNU开源的内存分析工具集，用于帮助开发者发现和调试程序中的内存错误、内存泄漏和性能问题。它在 Linux 系统上广泛使用，并提供了一系列工具，其中最常用的工具包括：Memcheck、Cachegrind、Callgrind、Massif、Helgrind、DRD DHAT、Experimental Tools

Valgrind 提供了一个强大的工具集，可用于诊断和优化 C/C++ 程序。开发者可以使用 Valgrind 来发现各种与内存管理、性能和多线程相关的问题，以改进程序的质量和性能

Valgrind 的最新版本是 valgrind-3.22.0（31 October 2023）。虽然 Valgrind 最常用于 Linux 系统上，但它也可以在其他平台上使用，Valgrind支持x86、x86-64、Armv7以及PowerPC上的Linux。除此之外，还有一些其它非正式支持的类Unix平台，比如FreeBSD、NetBSD 以及Mac OS X。需要注意的是，Valgrind 运行时会引入一定的性能开销，因此通常用于开发和调试阶段，而不是生产环境

Valgrind这个名字取自北欧神话中英灵殿的入口

## *Memcheck*

### 介绍

Memcheck：Memcheck 是 Valgrind 中最常用的工具，它用于检测内存错误和内存泄漏。Memcheck 能够跟踪程序运行时的内存分配和释放，并在发现以下问题时报告错误：

* 未初始化的内存读取
* 内存泄漏，即程序分配了内存但没有释放
* 重复释放已经释放的内存
* 内存越界访问
* 不匹配的内存分配和释放

### 使用

1. 在编译程序的时候打开调试模式，即gcc 编译器的 `-g` 选项，以便显示行号，编译时去掉 -O1、-O2 等优化选项;检查的是 C++程序的时候，考虑加上选项： `-fno-inline`，这样它函数调用链会很清晰

2. 执行

   ```cmd
   $ valgrind --tool=memcheck --leak-check=full --log-file=./log.txt ./YourProgram
   ```

   `-leak-check=full`：这个选项指定了内存泄漏检查的级别。`full` 选项会在程序退出时报告所有未释放的内存块，以帮助找到内存泄漏问题。其他可能的选项包括 `summary`（仅汇总内存泄漏信息）和 `yes`（默认级别，与 `full` 类似，但不会列出每个泄漏的详细信息）

   `--log-file=./log.txt`：这个选项指定了 `valgrind` 报告的输出文件的路径和名称

3. 程序运行结束，查看 log.txt 中的结果

### 结果分析

Memcheck包含下面这7类错误

1. illegal read/illegal write errors 非法读取/非法写入错误
2. use of uninitialised values 使用未初始化的区域
3. use of uninitialised or unaddressable values in system calls 系统调用时使用了未初始化或不可寻址的地址
4. illegal frees 非法的释放
5. when a heap block is freed with an inappropriate deallocation function 分配和释放函数不匹配
6. overlapping source and destination blocks 源和目的内存块重叠
7. memory leak detection 内存泄漏检测
   1. Still reachable 内存指针还在还有机会使用或者释放，指针指向的动态内存还没有被释放就退出了
   2. Definitely lost 确定的内存泄露，已经不能够访问这块内存
   3. Indirectly lost 指向该内存的指针都位于内存泄露处
   4. Possibly lost 可能的内存泄露，仍然存在某个指针能够访问某块内存，但该指针指向的已经不是该内存首位置
   5. Suppressed 某些库产生的错误不予以提示，这些错误会被统计到 suppressed 项目

## *Cachegrind*

Cachegrind 是一个性能分析工具，它可以模拟 CPU 缓存的行为，帮助开发者分析程序的缓存性能，找出瓶颈并进行优化，用来检查程序中缓存使用出现的问题

## *Callgrind*

Callgrind 是一个函数调用图分析工具，它可以帮助开发者了解程序中函数之间的调用关系、调用次数和执行时间，来检查程序中函数调用过程中出现的问题

## *Massif*

Massif：Massif 用于内存剖析，它可以跟踪程序的堆内存使用情况，帮助开发者分析内存分配和释放的模式，以及内存占用的变化

## *Helgrind*

* Helgrind 用于检测多线程程序中的数据竞态问题，它可以帮助开发者发现并调试线程同步问题，如死锁和竞争条件
* Helgrind 能够跟踪线程之间的同步操作，如互斥锁、条件变量等，以及共享数据的访问，以检测潜在的竞态条件。它会报告可能导致竞态条件的代码行，帮助开发者修复问题

## *DRD*

* DRD 也是 Valgrind 工具集中的一个工具，专门用于检测多线程程序中的数据竞态问题。与 Helgrind 不同，DRD 专注于数据竞态检测，不会执行其他 Valgrind 工具的分析任务。
* DRD 使用一种快速但有限的方法来检测数据竞态，因此通常比 Helgrind 更快。然而，由于其检测方法的局限性，它可能会漏报一些潜在的竞态条件。

## *Lackey & Nulgrind*

1. Lackey:
   * Lackey 是 Valgrind 工具集中的一个轻量级工具，用于帮助开发者调试程序中的多线程问题。它不像 Helgrind 或 DRD 那样检测数据竞态，而是通过记录并报告线程之间的线程交互来提供有关多线程程序行为的信息。
   * Lackey 会跟踪线程的创建、销毁和线程间的同步操作，如互斥锁和条件变量的使用。这些信息可以帮助开发者识别程序中的潜在问题，如死锁、线程间通信问题等。
2. Nulgrind:
   * Nulgrind 是 Valgrind 工具集中的一个伪工具，它实际上不执行任何内存或线程错误检测。相反，Nulgrind 用于测试和开发 Valgrind 自身。它提供了一种轻量级模式，用于加速 Valgrind 的自测试和调试过程，以便开发者更轻松地调试 Valgrind 工具集的其他部分。
   * Nulgrind 不会执行内存检查或线程检查，因此可以更快地运行，用于验证 Valgrind 的核心功能是否正常工作。

## *DHAT*

* DHAT 用于分析动态分配的堆内存，它不仅可以检测内存泄漏问题，还可以帮助开发者了解程序在堆内存使用方面的行为。
* DHAT 会跟踪堆内存分配和释放的情况，以及内存块之间的关系，从而帮助开发者发现潜在的内存泄漏和内存分配问题。DHAT 还提供了内存使用的可视化工具，以便更好地理解堆内存的使用模式。

## *Experimental Tools*

### BBV

### SGCheck

