MLIR（多级中间表示，Multi-Level Intermediate Representation 是 LLVM 原作者 Chris Lattner 在 Google 时候开始做的项目，现在已经合入LLVM仓库。MLIR
即Multi-Level Intermediate Representation,多级的中间表示。MLIR目的是做一个通用、可复
用的编译器框架,减少构建Domain Specific Compiler的开销。MILIR目前主要用于机器学习领
域,但设计上是通用的编译器框架,比如也有FLANG(Ilvm中的FORTRAN编译器),CIRCT(用
于硬件设计)等与ML无关的项目。MLIR现在还是早期阶段,还在快速更新迭代,发展趋势是尽可
能完善功能,减少新增自定义feature的工作量。



[MLIR介绍（一）概览 - 知乎](https://zhuanlan.zhihu.com/p/465464378)

[MLIR (llvm.org)](https://mlir.llvm.org/)

MLIR, Multi-Level Intermediate Representation 提供了一个用于构建和表示多个抽象级别计算的通用IR。MLIR 旨在解决高性能计算、机器学习、硬件设计等领域对复杂变换和优化的需求，并通过统一不同编程模型和硬件的表达来克服现有工具链的局限

MLIR（多级中间表示，Multi-Level Intermediate Representation）是由LLVM（低级虚拟机）社区开发的一种新的中间表示和基础框架。MLIR旨在解决现代计算中日益复杂的硬件和软件栈的需求。它最初由Google提出，主要用于支持TensorFlow的编译和优化，但它的设计目标远超这一应用场景，涵盖了广泛的应用。

MLIR的主要功能和优势包括：

1. **灵活的多级表示**：MLIR提供了一种将计算表示为多个级别的框架，从高度抽象的表示到接近底层硬件的表示。这种多级设计使得开发者能够在多个抽象级别上进行优化和变换。
2. **模块化和可重用性**：MLIR支持用户定义的方言（Dialect），允许开发者为特定的领域和应用定义自定义的操作和类型。这种模块化设计增强了代码可重用性，并促进了不同项目和团队之间的协作。
3. **可扩展性**：MLIR可以轻松扩展以支持新硬件、新优化和新语言。这使得它能够迅速适应新兴需求并集成前沿技术。
4. **性能优化**：通过支持跨多个层次的优化，MLIR可以实现比传统单层中间表示更高效的编译，从而提升性能。
5. **跨平台支持**：MLIR的设计考虑了多种目标平台，包括CPU、GPU和越来越多样化的专用加速器，这使得它非常适合异构计算环境。

MLIR的目标是统一高性能计算中的各种中间表示需求，简化编译器的构建和维护，使得针对特定领域的优化更为容易，并能更快适应未来技术的发展。它在机器学习编译、硬件加速器开发、图形渲染以及其他需要特殊优化的领域有着广泛的应用潜力

# Dialect

MLIR 支持的 Dialect 有

* Affine dialect
* Func dialect
* GPU dialect
* LLVM dialect
* SPIR-V dialect
* Vector dialect