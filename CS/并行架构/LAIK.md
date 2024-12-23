LAIK, **L**eichtgewichtige **A**nwendungs**I**ntegrierte Datenhaltungs**K**omponente, i.e. Lightweight ApplicationIntegrated Fault-Tolerant Data Container

LAIK的基本思想是提供一个轻量级的库，帮助HPC程序员实现（主动的）容错控制，基于数据迁移 Data Migration 的概念。该库包括一组API，以协助程序员动态分割和重新分割数据。通过将数据分割 Data Partitioning 的责任交给LAIK，LAIK可以根据当前的工作负载和硬件情况计算和重新计算最佳的分割选项。如果对即将发生的故障进行了预测，LAIK可以利用该信息进行重新分割。用户随后可以调整数据分配并从任何应用数据中释放故障节点

LAIK 通过接管 partitioning 的控制来协助采用SPMD模型编写的应用程序进行数据迁移。这样LAIK可以通过调整分区和排除故障节点，向用户应用程序提供关于预测到即将发生的故障的系统级信息

因此，应用程序可以通过采用新的分区方式来主动应对预测。 此外，如果程序员将应用程序数据的责任转移给LAIK，LAIK可以承担重新分区和重新分配数据的责任。 程序员指定一个分区器 partitioner 来告诉LAIK数据如何在不同的程序实例之间分布。这样，程序员只需要指定何时允许数据迁移，而重新分区和数据迁移的实际过程由LAIK自动控制完成。对于不同的目的和不同的抽象层（index space or data space），LAIK提供了不同的API

通过将数据分布的责任转移到LAIK，可以实现隐式通信 implicit communication。程序员只需要指定哪些数据的部分是需要的，以及何时需要。通过为同一数据结构提供多个分区，LAIK实现了这一点。通过改变当前活动的分区，这个过程称为 switch，数据可以在可用的程序实例之间重新分布，从而实现隐式通信。 自动数据分布和重新分配的另一个优势是可以确保在程序执行过程中实现负载平衡。随着未来出现的高性能计算系统，负载平衡变得至关重要以确保可扩展性。对于传统的高性能计算应用程序，通常通过显式实现自适应负载平衡。这导致了高度特定于应用程序的工程和实现，从而导致了高代码复杂性。这种高代码复杂性最终会导致可维护性问题。通过LAIK，这一责任也可以完全转移到LAIK，以确保用户代码的高模块化和低复杂性