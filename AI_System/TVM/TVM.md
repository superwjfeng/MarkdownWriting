# 介绍

## *架构*

TVM, Tensor Virtual Machine 张量虚拟机。利用 TVM，工程师可以在任意硬件后端高效地优化和运行计算

<img src="TVM架构.png">

从上图可以看出 TVM 架构的核心部分就是 NNVM 编译器。注意⚠️：最新的 TVM 已经将 NNVM 升级为了 Realy，选择上面这张图只是因为它比较清楚

## *安装*

### Docker

### 源代码编译







scheduler 可以简单理解为是一系列优化选择的集合，这些选择不会影响整个计算的结果，但对计算的性能却至关重要