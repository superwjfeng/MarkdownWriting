# OpenMP

## *intro*

现阶段很难做出自动并行的机器，其中难点在于：编译器不仅要弄清哪些操作之间是相互独立的，还要解决如何在并行环境中定位数据的棘手问题。编译器需要考虑的是整个代码，而非一个又一个的子程序。因此线程级并行需要程序员的参与。

共享内存中的一种并行方式是：程序员编写普通程序，并借助OpenMP向编译器发布何时并行或如何并行的指令。这种指明数据分布，并将并行问题留给编译器的方法是后面要介绍的 PGAS语言的基础。而在分布式内存 UMUA 中，程序员则需要通过MPI来管理一切

Open Multi-Processing, OpenMP 是C和Fortran的一个扩展包，它的主要并行方法是采用编译器指令来指示循环的并行执行。OpenMP主要包括编译器指令 compiler directives、库例程 library routine（例程指的是一些预定义的代码片段）以及一些环境变量 environment variables

OpenMP是基于线程的，它的特点是动态并行 dynamic parallelism：在代码的一个部分和另一个部分之间，并行运行的执行流的数量可以变化。**并行性是通过创建并行区域来声明的**，例如表明一个循环嵌套的所有迭代都是独立的，然后运行时系统将使用任何可用的资源

### OpenMP系统栈

### fork-join 模型

<img src="fork-join model.drawio.png" width="70%">

## *语法*

### demo

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void hello(void){
    int my_rank = omp_get_thread_num(); // 创建的线程组中的编号
    int thread_count = omp_get_num_threads(); // 创建的线程总数
    printf("Hello World from thread %d of %d.\n", my_rank, thread_count);
}

int main(int argc, char *argv[]) {
    int thread_count = strtol(argv[1], NULL, 10);
# pragma omp parallel num_threads(thread_count)
    hello();

    return 0;
}
```

引入 `<omp.h>` 库，编译的时候加上 `-fopenmp` 来开启OpenMP的编译器指令

### 编译器指令

所有的编译器指令以 `#paragma omp` 开头，比如parallel for的编译器指令为

```c
#pragma omp parallel for
```

* parallel：

* parallel for

  它告诉编译器接下来的for循环，将会使用并行的方式执行，也就是说把for的任务分配给不同的线程。使用并行的时候需要满足以下四个需求：

  * 在循环的迭代器必须是可计算的并且在执行前就需要确定迭代的次数

  * 在循环的代码块中不能包含break、return、exit

  * 在循环的代码块中不能使用goto跳出到循环外部

  * 迭代器只能够被for语句中的增量表达式所修改

  OpenMP编译器不检查被parallel for指令并行化的循环所包含的迭代间的依赖关系。若一个或者更多个迭代结果依赖于其他迭代的循环，一般不能被正确的并行化

* single

  * 告诉编译器接下来紧跟的下段代码将会由只一个线程执行
  * 它可能会在处理多段线程不安全代码时非常有用
  * 在不使用no wait选项时，在线程组中不执行single的线程们将会等待single的结束

* master

  * 它告诉编译器接下来的一段代码只会由主线程执行
  * 它不会出现等待现象，其他线程会直接完成，等到join

* section

  * sections在封闭代码的指定部分中，由线程组进行分配任务
  * 每个独立的section都需要在sections里面
    * 每个section都是被一个线程执行的
    * 不同的section可能执行不同的任务
    * 如果一个线程够快，该线程可能执行多个section

  ```c
  #pragma omp parallel shared(a,b,e,d) private(i)
  {
  #pragma omp sections
      {
  #pragma omp section
          {
              for (i = 0; i < N; i++) {
                  c[i] = a[i] + b[i];
              }
          }
  #pragma omp section
          {
              for (i = 0; i < N; i++) {
                  d[i] = a[i] * b[i];
              }
          }
      } /* end of sections */
  } /* end of parallel block */
  ```

* `reduction(op:list)`

### 同步互斥

* barrier
  * barrier是用于实现同步的一种手段。它会在代码的某个点，令线程停下直到所有的线程都到达该地方

* nowait：用于打断自动添加的barrier的类型，如parallel中的for以及single
* critical
* atomic

### 共享和私有变量

在并行区外的变量默认是共享变量 shared variables，可以被所有线程使用；私有变量 private variables 只能被一条线程操作

```c++
int i；
float *a, *b, *c, tmp；
#pragma omp parallel for private(tmp)
for (i = 0; i < N; i++) {
	tmp = a[i] / b[i];
    c[i] = tmp * tmp;
}
```

Data sharing clauses 数据共享语句有如下

* `private(var-list)`：在var-list中的变量是私有的

  * 每一个线程都独自拥有变量的副本
  * 如果j被定义为私有变量，那么在for循环里面所有的线程都不能访问其他的j，尽管j本来是共享变量
  * 所有的线程都不会使用到先前的定义，比如说一开始tmp若被赋值为3，则进入并行块后会被按照内置类型初始化为0.0
  * 所有线程都不能给共享的赋值
  * 在循环的入口以及出口，都不会进行定义，也就是说在并行块结束后不会对原来的变量产生影响

* `shared(var-list)`：在var-list中的是共享的

* `default(private | shared | none)`

  * 为一个并行区域的变量设置默认等级
  * 当设置为none的时候，若shared没有显式给出编译会报错

* `firstprivate(var-list)`：告诉编译器私有变量在**第一个循环**会继承共享变量的值，其他用法基本和private一样

* `lastprivate(var-list)`

  指定将线程中的一个或多个私有变量的值在并行处理结束后复制到主线程中的同名变量中，负责拷贝的线程是for或sections任务分担中的最后一个线程

### 循环调度策略

```c
#pragma omp for schedule(type[, size])
```

循环调度策略 loop scheduling strategies 有

* `static`
* `dynamic`
* `guided`
* `auto`
* `runtime`

### 常用API

* `omp_get_num_threads()`：返回并行域中总线程的数量
* `omp_get_thread_num()`：返回当前工作的线程编号
* `omp_set_num_threads()`：设定并行域内开辟线程的数量
* `omp_get_wtime()`：openmp中自带的计时函数，用于检测并行优化效果

### 环境变量

OpenMP提供了一些环境变量，用来在运行时对并行代码的执行进行控制。这些环境变量可以用来控制：设置线程数、指定循环如何划分、将线程绑定到处理器、启用/禁用嵌套并行，设置最大的嵌套并行级别、启用/禁用动态线程、设置线程堆栈大小、设置线程等待策略。常用的环境变量有：

* OMP_SCHEDULE：用于for循环并行化后的调度，它的值就是循环调度的类型
* OMP_NUM_THREADS：用于设置并行域中的线程数
* OMP_DYNAMIC：通过设定变量值，来确定是否允许动态设定并行域内的线程数
* OMP_NESTED：指出是否可以并行嵌套

## *OpenMP的内存模型*

# MPI

Message Passing Interface, MPI 消息传递接口库 顾名思义用于消息传递型的多处理器

MPI的接口大致可以分为

* 进程管理：包括查询并行环境和构建处理器的子集
* 点对点通信：一组用于两个进程之间进行交互的调用
* 集体调用：所有的处理器或者某个指定的处理器子集都会参与其中，比如广播调用

# CUDA

# OpenACC



