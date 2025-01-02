[MLIR (llvm.org)](https://mlir.llvm.org/)










EDSC, Embedded Domain Specific Constructs 嵌入式域特定构造：一个声明式的构造器库，用于以朴素 C++ API 的方式构造 MLIR 

Export 导出：一般用以说明从 MLIR 表示体系到其他语义等效的表示的操作，例如翻译 translation

Function 函数：一个有名称操作，它包含一个域 region

Import 导入：一般用以说明从其他表示到 MLIR 体系的语义等效表示的操作，例如翻译 translation





Module 单元：一个操作 operation，它包含一个域 region，这个域包含一个块 block，这个块由多个操作 operation 组成





Round-trip 往返：原表示到目标表示又回到原表示的过程

Terminator operation 终止操作：该操作 operation 终止一个块 block

Transitive lowering 传递下降：类似 `A->B->C` 的下降模式。通过部分转换 partial conversion 充分将非法操作合法化



# Architecture

## *传统编译器 IR 的问题*

### IR 碎片化问题

* LLVM IR 抽象级别太低，无法针对特定领域或者语言优化，因而各个语言框架有自己的 IR 和优化实现，无法重用
* IR 种类太多，针对不同种类 IR 的开发的 Pass 可能重复即不同 IR 的同类 Pass 不兼容。针对新的IR编写同类 Pass 需要重新学习 IR 语法，门槛过高
* 不同类型 IR 所做的 Pass 优化在下一层中不可见
* 不同类型 IR 间转换开销大，从图 IR 到 LLVM IR 直接转换存在较大开销

### MLIR 试图解决的问题

MLIR（多级中间表示，Multi-Level Intermediate Representation 是 LLVM 原作者 Chris Lattner 在 Google 时候开始做的项目，现在已经合入LLVM仓库。MLIR目的是做一个通用、可复用的编译器框架，减少构建 Domain Specific Compiler 的开销。MILIR 目前主要用于机器学习领域，但设计上是通用的编译器框架，比如也有 FLANG（Ilvm 中的 FORTRAN 编译器）、CIRCT（用于硬件设计）等与ML无关的项目。MLIR 现在还是早期阶段，还在快速更新迭代，发展趋势是尽可能完善功能，减少新增自定义 feature 的工作量

下面是一些用 MLIR 构建的开源项目

- tensorflow：没有 tf 就没有MLIR
- mhlo：tensorflow 组件，相当于支持动态规模的 XLA
- tfrt：tensorflow 组件，tensorflow 新的 runtime
- torch-mlir：连接 pytorch 与 mlir 生态
- onnx-mlir：连接 onnx 与 mlir 生态
- iree：深度学习 end2end 编译器
- circt：硬件设计及软硬件协同开发
- flang：FORTRAN 的编译器前端
- polygeist：C/C++ source code 变成 mlir Affine

### 与 LLVM IR 的区别 & 联系

[MLIR介绍（一）概览 - 知乎](https://zhuanlan.zhihu.com/p/465464378)



> 个人认为MLIR更适合和LLVM做比较,而不是TVM等dl compiler。LLVM和MLIR的很多概念都比
> 较像,了解LLVM的话MLIR会比较容易上手。
> LLVMIR由于当时的历史局限性,类型只设计了标量和定长vector,有个给LLVM加matrix类型的
> 提案目前看来也没有进展。而MLIR自带tensor*类型,对深度学习领域更友好。
> MLIR有Operation和Dialect的概念,Dialect,Operation,Attrribute,Type等都可以通过td文
> 件比较方便地定义出来。而LLVM定义新的intrinsic比较麻烦,定义新的IR就更麻烦了。LLVMIR
> 主要表示硬件指令操作,而MLIR能表示更多东西,比如表示神经网络的图结构。因为有Dialect,
> MLIR是组件化+,去中心的,不像LLVM的ir是一种大而全的。
> MLIR执行过程和LLVM一样,IR会过由Pass组成的Pipeline,不所地变换生成最终的IR。不同的是
> MLIR的IR可以是不同dialect的,构成了Multi-Level的效果。

## *Operation*

Operation 是 Dialect 的重要组成部分，可以看作是方言语义的基本元素

### Recursive Nesting Architecuture of Operation

<img src="Dialect结构.drawio.png">

* 
  Op, Operation 操作：表示一个代码单元。是MLIR最重要的概念之一。`Op` 是 `operation*` 的 wrapper
* Operation 的结构是一个嵌套递归结构，即 Operation `->` Region `->` Block `->` Operation `->` `...`
* Region 域：为多个 Block 的控制流图 CFG/列表
* Block 块：一个多个不含控制流 control flow 的 Operations 组成的顺序表

### Operation 的格式

和 LLVM IR 以 SSA instruciton 为核心不同，MLIR 没有预定义的 instruction，全部都是 operation

BTW，不同 dialect 之间统一的 MLIR 形式也是 dialect conversion 高效的原因之一

<img src="OperationFormat.png">

MLIR 的格式类似于 LLVM IR，都是基于 SSA 的

Operation 看起来就像一个函数的定义，有输入输出

上面的完整格式可能看起来有些复杂，一般的 Operation 的格式为

```
%result = "dialect.operation_name"(%arg1, %arg2) : (type1, type2) -> type3
```

## *Dialect*

### 什么是 dialect

Dialect 是 MLIR 的核心机制，它其实就代表了一层层的 IR（也可以理解为对一门 DSL 的抽象）。Dialect 是一系列用以拓展 MLIR 体系的组件，其于 MLIR 类似于各种库之于 C++。不同的方言以不同的名字空间 namespace 体现

将多个层次的 IR 通过 Dialect 方言机制进行语义的统一，共用同一套生态系统，可以使各个层次之间的跨度缩小，从而有效地实现各层次之间地协调优化

原来多种后端对应多个中间 IR，现在可以通过 dialect 之间的互相转换，只需要 graphIR 一个

### Dialect 的主要组成

* A prefix (“namespace” reservation) "命名空间" 前缀
* (optional) A list of custom types, each its C++ class 一个独有的类型系统
* A list of operations, each its name and C++ class implementation
  * Verifier for operation invariants (e.g. toy.print must have a single operand)
  * Semantics (has-no-side-effects, constant-folding, CSE-allowed, ….)
* Passes: analysis, transformations, and dialect conversions. 分析、转换
* (optional) Possibly custom parser and assembly printer 针对当前 dialect IR 自定义的解析器、打印器



# 构建 Dialect

<img src="Operation架构.drawio.png">

构建 operation 采用了一种声明式的自动化工具 ODS, Operation Definition Specification：基于 TableGen，方便自定义 operation

使用 mlir-tablegen 工具从 `.td` 文件转换为 `.inc` 文件

## *OpTrait*

`OpTrait` 是用于表达 operation 属性和行为的一种机制。它们是一组模板类，能够以一种可组合的方式为操作添加特定的特征或者约束。通过使用 `OpTrait`，开发者可以简化操作的定义，避免重复代码，并使得操作的行为更具一致性和可预测性

### Built-in OpTrait

* `ZeroOperands` 和 `OneResult` 等：用于指定操作的操作数数量或结果数量。例如，`ZeroOperands` 表示操作没有操作数，而 `OneResult` 表示操作有一个结果
* `NOperands` 和 `NResults`：用于指定确切数量的操作数和结果，自定义操作中可以直接使用
* `SameOperandsAndResultType`：表示所有操作数和结果类型必须相同，这在很多数学操作中常见
* `HasNoSideEffect`：表示操作没有副作用，即操作的执行不影响程序的状态，适合用在纯函数或者纯计算的操作中
* `IsCommutative` 和 `IsIdempotent`：定义操作的数学性质，比如交换律和幂等律，这对于优化和变换很有帮助
* `AffineScope`：用于标记操作符为仿射作用域的一部分，适用于仿射表达式和仿射循环的上下文

### 自定义 OpTrait

除了使用内置的 `OpTrait`，开发者还可以实现自定义的 `OpTrait` 以满足特定需求。实现自定义 `OpTrait` 涉及到定义接口类以及实现检查逻辑，通常是在模板类中实现的

以下是一个示例，展示如何定义一个简单的自定义 `OpTrait`，例如要求所有操作数的类型必须是整数类型：

```c++
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class AllOperandsIntegers : public OpTrait::TraitBase<ConcreteType, AllOperandsIntegers> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    for (auto operand : op->getOperands()) {
      if (!operand.getType().isIntOrIndex()) {
        return op->emitOpError("requires all operands to be integer or index type");
      }
    }
    return success();
  }
};

} // namespace OpTrait
} // namespace mlir
```

在这个例子中，我们定义了一个 `AllOperandsIntegers` 特征，检查操作中的所有操作数是否为整数或索引类型。这个特征可以在操作定义中通过模板参数轻松使用

### 在 TableGen 中使用 OpTrait

在 TableGen 操作定义中，可以在操作的 trait 列表中使用这些 traits。一个简单的例子如下：

```
def MyOperation : MyDialect<"my_op">,
                  Commutative,
                  Pure,
                  OpTrait::ZeroRegion,
                  OpTrait::NOperands<2>::Impl,
                  OpTrait::OneResult {
  let summary = "My custom operation with traits";
  let description = [{
    This operation is commutative, has no side effects, and
    operates on two operands returning one result.
  }];
}
```

这里，`MyOperation` 持有多个 traits，如可交换性、纯运算、没有区域（没有控制流子图），两个操作数和一个结果。这使得操作的定义精确且易于阅读，同时实现了良好的属性检查和优化支持

## *OpInterface*

OpInterface 是一套提供协议和方法的机制，允许为 operation 定义统一的接口。这些接口定义了一组方法，操作可以选择实现这些方法以提供某些行为或属性。这种机制促进了代码的可重用性和多态性，使得不同类型的操作能够以统一的方式进行处理和转换

# Dialect 转换

* Translation 翻译：用以区分转换 conversion，翻译是**非 MLIR 表示和 MLIR 表示之间**的操作，比如说高级语言（C、XLA、Toy、深度学习框架等）在接入 MLIR 之前需要做一步翻译操作
* Conversion 转换：用以区分翻译 translation，转换是 **MLIR 不同方言之间**的语义等效转换操作
* Transformation
* Canonicalization 正规化
* Lowering 下降 ：表示从高级别 IR 表示到低级别 IR 表示，语义等效表示的改变。MLIR 中通常指转换 conversion

和实现 operation 的 ODR 风格类似，实现 dialect 转换的方式是 DRR, Declarative Rewrite Rule 声明式重写规则：基于 TableGen，通过编写的声明式的重写规则，可以自动生成表达式匹配和重写函数，即生成等效的 C++ 的 `mlir::RewritePattern` 子类

## *Conversion*

## *Lowering 过程*

MLIR 是没有可以生成目标代码的 codegen 的，所以必须要将 IR 转换为 LLVM IR，这个过程称为 Lowering，属于 dialect 转换的一种

这里先引入一个 Legalization 合法化 的概念：改变当前 operation 的表示以符合 conversion target 要求

<img src="Lowering过程.drawio.png" width="80%">

### Components

* Conversion Target: Specification of what operations are legal and under what circunnstances
* Operation Conversion: Dag-Dag patterns specifying how to transform illegal operations to legal ones
* Type Conversion 类型转换匹配: Specification of how to transform illegal types to legal ones

### Lowering Modes

有两种 lowering 的模式

* Partial: Not all input operations have to be legalized to the target。这是 MLIR 比较特色的转换方式， 可以保留原 dialect 中仍然需要的 operation，不像 LLVM IR 转换就把原来的 IR 全部转换走了
* Full: All input operations have to be legalized to the target

### Partial Lowering

### Full Lowering

1. 创建 Lower to LLVM Pass
2. Type Conversion
3. Conversion Target
4. 定义 lowering pattern



# Built-in Dialect

## *Overview*

既然 Dialect 的作用是类似于各种库之于 C++，那么 built-in dialect 就类似于 C++ 的标准库

MLIR 原生支持的内建 Dialect 有很多，具体可以查看 [Builtin Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/Builtin/)，我们下面只介绍几个最常用的 built-in dialect

* Affine dialect：处理循环嵌套，实现了循环展开、多面体变换等一些算法
* Func dialect：处理函数的 dialect，包含的函数定义、调用、返回等基本操作
* Arith dialect：处理加减乘除移位等各种运算
* Math dialect：
* SCF, Standard Control Flow dialect：结构化控制流，保留 for，if 等语句
* CF, Control Flow dialect：
* Vector dialect：de
* GPU dialect：
* LLVM dialect：LLVM IR 的 binding，可以直接翻译给 LLVM 做后续编译
* SPIR-V dialect

### Standard dialect 的拆分

Standard dialect 已被拆分和重新组织到更为专注的方言中。这种重构有助于更清晰地表述操作的用途和领域范围，并且加强模块化和可扩展性。以下是对一些关键变化的概述：

* arith Dialect
  - 这部分承担了几乎所有整数和浮点数的算术及比较操作
  - 例子包括 `arith.addi`（整数加法）、`arith.subf`（浮点数减法）、`arith.cmpi`（整数比较）、`arith.cmpf`（浮点数比较）等
* func Dialect ['func' Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/Func/)
  - 用于处理函数定义及相关操作，如函数调用、参数传递和返回
  - `func.func`, `func.return` 等是该方言的典型操作
* memref Dialect
  - 处理内存相关的操作，包括分配、释放、加载和存储等动态内存操作
  - `memref.alloc`, `memref.dealloc`, `memref.load`, `memref.store` 都属于此类
* cf Dialect（Control Flow）
  - 专注于流控制相关的操作，如条件分支、循环控制等
  - `cf.br`, `cf.cond_br` 是该方言的基本操作
* tensor 和 vector Dialects
  - 这些方言分别处理张量和向量相关的操作，是高性能计算、机器学习等领域中的重要组成部分
  - 包含 `tensor.extract`, `tensor.insert` 以及 `vector.add`, `vector.mul` 等操作

### Architecture of Dialect

[【源码研读】MLIR Dialect 分层设计 - Aurelius84 - 博客园](https://www.cnblogs.com/CocoML/p/17632342.html)

<img src="Dialect类型分类图.png">

## *linalg*



## *Affine*

['affine' Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/Affine/#dimensions-and-symbols)

## *LLVM IR*

## *memref*





# Type & Arrtibute

[Defining Dialect Attributes and Types - MLIR](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)

* Type：MLIR 中任何数据都必须指定 Type；MLIR 中内置了很多常用的 Type，我们也可以拓展自己的Type，来表示更复杂的数据类型
* Attribute：MLIR 中 Attribute 可以简单理解为 Constant 常量数据值，用来定义一些常量和属性。每个 Attribute都有其 Type

## *Type*

## *Attribute*

> Attributes are the mechanism for specifying constant data on operations in places where a variable is never allowed - e.g. the comparison predicate of a [`cmpi` operation](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpi-arithcmpiop). Each operation has an attribute dictionary, which associates a set of attribute names to attribute values. MLIR’s builtin dialect provides a rich set of [builtin attribute values](https://mlir.llvm.org/docs/LangRef/#builtin-attribute-values) out of the box (such as arrays, dictionaries, strings, etc.). Additionally, dialects can define their own [dialect attribute values](https://mlir.llvm.org/docs/LangRef/#dialect-attribute-values).

# Pass

### Pass 分类

* 按场景分
* 最常用的 Pass：模式匹配并变换 pattern match & rewrite
* 验证 Pass：借助 llvm-lit & FileCheck
* 多线程运行 Pass

## *两种遍历 IR 的方式*

### 适合使用裸 `walk()` 的场景

### 适合使用 Pattern 的场景

## *Pattern*

### PatternSet

PatternSet 是一组 Patterns，在多个 Passes 间共享

## *PassManager*

# Support

和 Clang 中的 Support 作用一样，就是封装 MLIR 对 OS 系统调用接口的使用，从而提供跨平台的支持



# Polly

Polly 是 LLVM 项目的一个子项目，它提供了自动并行化和循环优化的功能。Polly 使用高级多维数组索引（Affine Expressions）来理解、表示和优化循环嵌套，特别是那些对于性能至关重要的计算密集型循环

Polly 基于一种叫做多面体模型的数学表示，使用这种方法，可以进行复杂的优化

Polly 主要应用于需要大规模数值计算的科学和工程领域，例如物理模拟、矩阵运算和图像处理。在这些领域，循环结构往往占据了程序的绝大部分计算时间，并且有明确的数据依赖模式可供分析和优化