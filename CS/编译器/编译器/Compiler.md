# Introduction

## *编译器构造*

<img src="三段优化编译器结构.drawio.png" width="80%">

### 前端

前端负责理解源语言程序

* 词法分析 Lexical Analysis (Parser)：扫描源代码字符流，并将其组织成有意义的词法单元 token 序列
* 语法分析 Syntax Analysis：语法分析器检查源代码的语法结构是否符合编程语言的规则，并按照语法规则将代码组织成树状结构（比如 抽象语法树 Abstract Syntax Tree, AST）
* 语义分析 Semantic Analysis：检查变量的声明、类型匹配、函数调用等，以确保程序在逻辑上是合理的

### 优化/中端

优化器分析代码的的IR形式，通过发现有关上下文的事实来重写代码，以求提高实现的效率

理想情况下优化器应该是机器无关的，即它应该通用于所有的机器。但事实并非如此，优化器总是要用到一些机器特性

### 后端

后端负责将程序映射到目标机器上

* 指令选择 Instruction Selection：选择适当的目标机器指令来执行高级语言中对应的操作，同时考虑目标机器的特定约束和优化策略
* 指令调度 Instruction Scheduling：通过重排指令 reordering 的执行顺序，指令调度旨在减少指令之间的依赖关系，最大程度地利用目标机器的硬件资源，从而提高程序的性能
* 寄存器分配 Register Allocation：最大限度地减少内存访问次数，以提高程序的执行速度

## *ILOC*

ILOC, Intermediate Language for an Optimizing Compiler 是一种用于优化编译器的中间表示 IR

ILOC 的设计旨在简化编译器的分析和优化任务，同时提供足够的灵活性，可以说它是一种RISC的核心指令集

<img src="ILOC.drawio.png">

### 命名规则

1. 变量的内存偏移量表示为变量名加前缀 `@` 字符
2. 用户可以假定寄存器数量是无限的，它们用简单的整数或符号名引用，前者如 `r_1776`，后者如 `r_i`
3. 寄存器rarp是保留的，用做指向当前活动记录的指针

## *编译器与NLP的对比*

https://cloud.tencent.com/developer/article/1776822

# Scanning



# Optimization

Common Subexpression Elimination, CSE

# Backend

