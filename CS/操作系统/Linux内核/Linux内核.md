# 源代码

## *代码结构*

* arch：包含和硬件体系结构相关的代码，每种平台占一个相应的目录。比如说和ARM体系存放在arm目录下，其中包括kerel（内核核心部分）、mm（内存管理）、nwfpe（浮点单元仿真）、lib（硬件相关工具函数）、boot（引导程序）、configs（各开发板的默认配置文件所在）
* block：部分块设备驱动程序
* crypto：常用加密和散列算法（如AES、SHA等），还有一些压缩和CRC校验算法
* Documentation：关于内核各部分的参考文档
* drivers：设备驱动程序，一般每种不同的设备的驱动程序占用一个子目录
* fs：各种支持的文件系统的相关代码，如ext、fat、ntfs等
* include：这个目录包括了核心的大多数include文件。另外对于每种支持的体系结构分别有一个子目录
* init：内核初始化代码（注意不是系统引导代码）
* ipc：此目录包含了核心的进程间通讯代码
* kernel：内核的最核心部分，包括进程调度、定时器等，和平台相关的一部分代码放在 `arch/*/kernel` 目录下
* lib：此目录包含了核心的库代码。与处理器结构相关库代码被放在 `arch/*/ib/` 目录下
* mm：内存管理代码，和平台相关的一部分代码放在 `arch/*/mm` 目录下
* net：网络相关代码，实现了各种常见的网络协议
* scripts：用于配置内核的脚本文件
* security：主要是一个SELinux的模块
* sound：常用音频设备的驱动程序等
* usr：目前实现了一个cpio

# 进程 & 线程

## *task_struct*

<img src="task_struct.png">

```c
struct task_struct {
    volatile long state;  //说明了该进程是否可以执行,还是可中断等信息
    unsigned long flags;  //Flage 是进程号,在调用fork()时给出
    int sigpending;       //进程上是否有待处理的信号
    mm_segment_t addr_limit; //进程地址空间,区分内核进程与普通进程在内存存放的位置不同
                             //0-0xBFFFFFFF for user-thead
                             //0-0xFFFFFFFF for kernel-thread
    //调度标志,表示该进程是否需要重新调度,若非0,则当从内核态返回到用户态,会发生调度
    volatile long need_resched;
    int lock_depth;  //锁深度
    long nice;       //进程的基本时间片
    //进程的调度策略有三种：实时进程:SCHED_FIFO, SCHED_RR, 分时进程:SCHED_OTHER
    unsigned long policy;
    struct mm_struct *mm; //进程内存管理信息
    int processor;
    //若进程不在任何CPU上运行, cpus_runnable 的值是0，否则是1 这个值在运行队列被锁时更新
    unsigned long cpus_runnable, cpus_allowed;
    struct list_head run_list; //指向运行队列的指针
    unsigned long sleep_time;  //进程的睡眠时间
    //用于将系统中所有的进程连成一个双向循环链表, 其根是init_task
    struct task_struct *next_task, *prev_task;
    struct mm_struct *active_mm;
    struct list_head local_pages;       //指向本地页面      
    unsigned int allocation_order, nr_local_pages;
    struct linux_binfmt *binfmt;  //进程所运行的可执行文件的格式
    int exit_code, exit_signal;
    int pdeath_signal;     //父进程终止时向子进程发送的信号
    unsigned long personality;
    //Linux可以运行由其他UNIX操作系统生成的符合iBCS2标准的程序
    int did_exec:1; 
    pid_t pid;    //进程标识符，用来代表一个进程
    pid_t pgrp;   //进程组标识，表示进程所属的进程组
    pid_t tty_old_pgrp;  //进程控制终端所在的组标识
    pid_t session;  //进程的会话标识
    pid_t tgid;
    int leader;     //表示进程是否为会话主管
    struct task_struct *p_opptr,*p_pptr,*p_cptr,*p_ysptr,*p_osptr;
    struct list_head thread_group;   //线程链表
    struct task_struct *pidhash_next; //用于将进程链入HASH表
    struct task_struct **pidhash_pprev;
    wait_queue_head_t wait_chldexit;  //供wait4()使用
    struct completion *vfork_done;  //供vfork() 使用
    unsigned long rt_priority; //实时优先级，用它计算实时进程调度时的weight值

    //it_real_value，it_real_incr用于REAL定时器，单位为jiffies, 系统根据it_real_value
    //设置定时器的第一个终止时间. 在定时器到期时，向进程发送SIGALRM信号，同时根据
    //it_real_incr重置终止时间，it_prof_value，it_prof_incr用于Profile定时器，单位为jiffies。
    //当进程运行时，不管在何种状态下，每个tick都使it_prof_value值减一，当减到0时，向进程发送
    //信号SIGPROF，并根据it_prof_incr重置时间.
    //it_virt_value，it_virt_value用于Virtual定时器，单位为jiffies。当进程运行时，不管在何种
    //状态下，每个tick都使it_virt_value值减一当减到0时，向进程发送信号SIGVTALRM，根据
    //it_virt_incr重置初值。
    unsigned long it_real_value, it_prof_value, it_virt_value;
    unsigned long it_real_incr, it_prof_incr, it_virt_value;
    struct timer_list real_timer;   //指向实时定时器的指针
    struct tms times;      //记录进程消耗的时间
    unsigned long start_time;  //进程创建的时间
    //记录进程在每个CPU上所消耗的用户态时间和核心态时间
    long per_cpu_utime[NR_CPUS], per_cpu_stime[NR_CPUS]; 
    //内存缺页和交换信息:
    //min_flt, maj_flt累计进程的次缺页数（Copy on　Write页和匿名页）和主缺页数（从映射文件或交换
    //设备读入的页面数）； nswap记录进程累计换出的页面数，即写到交换设备上的页面数。
    //cmin_flt, cmaj_flt, cnswap记录本进程为祖先的所有子孙进程的累计次缺页数，主缺页数和换出页面数。
    //在父进程回收终止的子进程时，父进程会将子进程的这些信息累计到自己结构的这些域中
    unsigned long min_flt, maj_flt, nswap, cmin_flt, cmaj_flt, cnswap;
    int swappable:1; //表示进程的虚拟地址空间是否允许换出
    //进程认证信息
    //uid,gid为运行该进程的用户的用户标识符和组标识符，通常是进程创建者的uid，gid
    //euid，egid为有效uid,gid
    //fsuid，fsgid为文件系统uid,gid，这两个ID号通常与有效uid,gid相等，在检查对于文件
    //系统的访问权限时使用他们。
    //suid，sgid为备份uid,gid
    uid_t uid,euid,suid,fsuid;
    gid_t gid,egid,sgid,fsgid;
    int ngroups; //记录进程在多少个用户组中
    gid_t groups[NGROUPS]; //记录进程所在的组
    //进程的权能，分别是有效位集合，继承位集合，允许位集合
    kernel_cap_t cap_effective, cap_inheritable, cap_permitted;
    int keep_capabilities:1;
    struct user_struct *user;
    struct rlimit rlim[RLIM_NLIMITS];  //与进程相关的资源限制信息
    unsigned short used_math;   //是否使用FPU
    char comm[16];   //进程正在运行的可执行文件名
     //文件系统信息
    int link_count, total_link_count;
    //NULL if no tty 进程所在的控制终端，如果不需要控制终端，则该指针为空
    struct tty_struct *tty;
    unsigned int locks;
    //进程间通信信息
    struct sem_undo *semundo;      //进程在信号灯上的所有undo操作
    struct sem_queue *semsleeping; //当进程因为信号灯操作而挂起时，他在该队列中记录等待的操作
    //进程的CPU状态，切换时，要保存到停止进程的task_struct中
    struct thread_struct thread;
     
    struct fs_struct *fs;        //文件系统信息
    struct files_struct *files;  //打开文件信息
      
    spinlock_t sigmask_lock;     //信号处理函数
    struct signal_struct *sig;   //信号处理函数
    sigset_t blocked;            //进程当前要阻塞的信号，每个信号对应一位
    struct sigpending pending;   //进程上是否有待处理的信号
    unsigned long sas_ss_sp;
    size_t sas_ss_size;
    int (*notifier)(void *priv);
    void *notifier_data;
    sigset_t *notifier_mask;
    u32 parent_exec_id;
    u32 self_exec_id;

    spinlock_t alloc_lock;
    void *journal_info;
};
```

Linux中的 `task_struct` 类型的结构体是进程描述符 process descriptor ，用来组织、管理进程资源

## *进程状态*

### task_struct中的state

<img src="Linux进程状态转移图.png" width="80%">

进程状态查看：`ps aux 或 ps axj`。Linux 2.6 内核的定义

<img src="PCB_kernel_def.png">

* TASK_RUNNING / R运行状态 Running：并不意味着进程一定在运行中，它表示进程要么是在运行中要么在**CPU运行队列**里排队

  <img src="runningStatus.png">

* TASK_INTERRUPTIBLE / S睡眠状态 Sleeping：意味着进程在等待睡眠完成（这里的睡眠也可叫做可中断睡眠 **interruptible sleep**），S状态对应的理论状态为阻塞态和挂起态，在等待非CPU资源就位，或者说在**非CPU硬件的队列**里排队。当等待的资源就位后，产生一个硬件中断或信号来环境进程。sOS可以通过调度算法在内存不够时将进程换出到Swap区

  * 情况一

    <img src="sleepingStatusSitu1.png">

    * 由于CPU的运行速度极快，实际上在该死循环中只有极少的时间在运行，绝大多数时间都在睡眠状态，将CPU资源给其他程序使用
    * 注意：当进程状态有一个 `+` 时，表示该任务为前台进程。具体可以看 *系统编程.md* 前台进程与后台进程的区别

  * 情况二

    <img src="sleepingStatusSitu2.png">

    一直在等待用户输入，所以一直处于IO的队列中，处于睡眠状态

* TASK_UNINTERRUPTIBLE / D磁盘休眠状态 Disk sleep：也可叫做深度睡眠/不可中断睡眠 **uninterruptible sleep**，不可以被被动唤醒，在这个状态的进程通常会等待IO的结束

  * 例子：一个进程正在往硬盘或者往其他IO设备写入数据，但此时该进程仍然占用了内存资源。若此时OS压力过大，可能会选择终止处于S状态的进程以保护整体的OS。当进程处于D状态时，则不能被OS终止，只能等该进程结束读写后自动醒来时，OS再结束它
  * D状态一般用于硬盘的读写，因为涉及到用户的数据比较重要
  * D状态的模拟代码不给出，因为要模拟这个进程需要大量数据的IO读写

* TASK_STOPPED / T暂停状态 Stopped

  * 当进程接收到 SIGSTOP、SIGTSTP、SIGTTIN、SIGTTOU 信号后进入暂停状态，可以通过 `kill -19 [PID]` 暂停进程
  * 和D状态、S状态相比，前两者都在等待某项资源或执行任务，但T状态并没有，是被用户手动暂停的
  * 调试时会呈现T状态，比如gdb打断点时gdb会给进程发送 `kill -19 [PID]` 暂停进程

* TASK_TRACED / T tracing stop 调试状态：进程的执行由debugger程序暂停


下面的两个进程状态既可以放在task_struct的state字段里，也可以放在exit_state字段里。只有当进程的执行被终止时，进程的状态才会变为这两种状态中的一种

* EXIT_ZOMBIE / Z僵死状态 Zombie，具体见下

* EXIT_DEAD / X死亡状态 Dead：这个状态只是一个返回状态，不会在任务列表里看到这个状态。和理论中的新建状态类似，存在时间极短，资源立刻会被OS回收

### 僵尸进程

* 什么是僵尸进程 Zombie Process：一个进程已经退出，但是还不允许被释放，处于一个被检测的状态。进程一般尤其父进程或者init检测回收，维持该状态是为了让父进程和init来回收其返回或产生的数据（回收就是由Z状态转到X状态）

* 为什么会产生僵尸进程：子进程已退出，父进程还在运行，但父进程没有读取子进程状态，子进程就会变成Z状态

  ```c
  #include <stdio.h>
  #include <stdlib.h>
  #include <unistd.h>                                                        
                      
  int main() {
      pid_t id = fork();
      if (id<0) {
          perror("fork");
          return 1;
      } else if (id == 0) {
          // Child process
          while (1) {
              printf("I am child, pid: %d, ppid:%d\n", getpid(), getppid());
              sleep(3);
              break;
          }
          exit(0); // 子进程直接退出
      } else {
          // Parent process 
          while (1) {
              printf("I am father, pid: %d, ppid:%d\n", getpid(), getppid());
              sleep(1);
          }
      }                           
      printf("You can see me!\n");
      sleep(1);
  
      return 0;
  }   
  ```

  <img src="ZombieProcess.png" width="80%">

  `<defunct>` 指的是父子进程间失去通信的进程

* 僵尸进程的危害

  * 若父进程或OS一直不读取子进程的返回，那么子进程就一直处于Z状态
  * 维护退出状态本身就是要用数据维护，也属于进程的基本信息，所有也保留在PCB中，若一直处于Z状态，则一直要维护PCB
  * 进程需要占据内存资源，若一直不回收，会造成内存浪费，也就是内存泄漏

* 孤儿进程 Orphan Process

  * 孤儿进程和僵尸进程相反，若父进程先于子进程结束，那么子进程变成Z进程后就成为孤儿进程

  * 孤儿进程被1号init进程领养后由init进程回收，下图可看到，PID为7425的子进程在PID为15713的父进程退出后被1号init进程领养了

    <img src="OrphanProcess.png">

## *链表管理*

Linux内核定义了 `list_head` 的带头双向循环链表的数据结构

### 进程链表

task_struct 有一个tasks链表

```c
struct list_head tasks {
    struct list_head *prev;
    struct list_head *next;
};
```

<img src="tasks链表.drawio.png" width="70%">

进程链表的哨兵位头是init_task，它就是0进程 process 0或swapper进程

### 进程间关系

<img src="进程间的亲缘关系.png" width="40%">

* real_parent
* parent
* children
* sibling

进程1（init）是所有进程的祖先

### pidhash & 链表

## *进程组织*

### TASK_RUNNING

### 等待队列

## *进程切换*

### 理解 `pthread_t` 线程id

OS本身并不提供线程的系统调用接口是因为线程的全部实现并没有全部体现在OS内，而是OS提供服务流，具体的线程结构由用户层的库来管理。要进行管理就需要在线程库中维护相关的数据结构

主线程的独立栈结构，用的是地址空间中的栈区；**而新线程用的栈结构，用的是线程库中维护的栈结构**

<img src="线程库数据结构.png" width="80%">

```c
struct thread_info {
	pthread_t tid;
    void *stack; //私有栈
}
```

### `tid` 与 `lwp` 标识线程身份

`tid` 是用户级别的，可以由 `pthread_self()` 得到

`lwp` 是内核级别的，需要使用 `syscall(SYS_gettid)` 来获取

```cpp
void *startRoutine(void *args) {
    while (true) {
        cout << "thread: " << pthread_self() << " |global_value: " << global_value << 
            " |&global_value: " << &global_value << " |Inc: " << global_value++ <<
            " |lwp: " << syscall(SYS_gettid) << endl;
        sleep(1);
    }
}
```

<img src="用户级tid和内核级lwp.png">

## *内核线程*

# Paging of x86-64

https://zhuanlan.zhihu.com/p/652983618 & Intel® 64 and IA-32 Architectures Software Developer Manuals Volume 3 Chapter 4 Paging

## *分页模式*

### 开启分页

分页 Paging 只能在保护模式（CR0.PE = 1）下使用。在保护模式下是否开启分页是由 CR0. PG 位（31位）决定的

* 当 CR0.PG = 0 时，未开启分页，此时线性地址等同于物理地址
* 当 CR0.PG = 1 时，开启分页

### 四种分页模式

intel-64 处理器支持 4 种分页模式

* 32 位分页
* PAE, Physical Address Extension 分页
* 4 级分页
* 5 级分页

处理器当前处于哪种分页模式是由 CR4.PAE、CR4.LA57 以及 IA32_EFER.LME 共同决定的

* 若 CR4.PAE = 0， 使用的是 **32位分页** 模式
* 若 CR4.PAE = 1 且 IA32_EFER.LME = 0，使用的是 **PAE 分页**模式
* 若 CR4.PAE = 1， IA32_EFER.LME = 1 且 CR4.LA57 = 0，使用的是 **4 级分页**模式
* 若 CR4.PAE = 1， IA32_EFER.LME = 1 且 CR4.LA57 = 1，使用的是 **5 级分页**模式

## *4级分页详解*

page-map level-4, PML4

Control Register 3, CR3

# 内存寻址

## *内存地址总览*

具体可以看 *操作系统理论.md*

### 三种地址

* 逻辑地址 logical address：包含在机器语言指令中用来指令一个操作数或一条指令的地址。每一个逻辑地址都由一个段 segment 和偏移量 offset 组成
* 虚拟地址 virtual address 或线性地址 linear address：32位或64位的连续无符号整数表示的虚拟地址
* 物理地址 physical address：用于memory chip的内存单元寻址，物理地址与从CPU的地址引脚发送到memory总线上的电信号相对应，物理地址也由32位或64位的连续无符号整数表示

### 地址翻译

<img src="地址翻译.drawio.png">

MMU中的TWU负责两部分寻址

* 分段单元 segmentation unit 硬件电路：把一个逻辑地址转换成虚拟地址
* 分页单元 paging unit 硬件电路：把一个虚拟地址转换为物理地址

## *硬件分段*

## *Linux分段机制*

### Linux GDT

### Linux LDT

## *Linux分页机制*

x86-64的物理分页见上

从 Linux 的早期版本开始，Linux就支持两级分页；Linux 2.6版本开始支持PAE和三级分页，从 Linux 2.6.11开始支持四级分页，Linux 4.11 版本引入了对五级分页的支持。64位系统到底是用三级、四级还是五级分页取决于具体硬件对线性地址/虚拟地址的划分



* 全局页目录 PGD Page Global Director
* 上层页目录 PUD Page Upper Directory
* 中间页目录 PMD Page Middle Directory
* 页表项 PTE Page Table Entry

# 内存管理

## *伙伴系统管理物理内存*

伙伴系统 Buddy System

## *slab分配器*

# 虚拟内存

# 锁 & 进程间通信

## *内核锁*

### RCU

RCU, Read-Copy-Update 是一种用于实现并发数据访问的机制，在 2.5 版本引入，通常用于多线程环境下对共享数据的访问和更新。它的设计目标是允许读操作可以在不加锁的情况下并发进行，而不会阻塞写操作

它进行并发读写的过程如下：

1. 操作/读 read：读操作可以在不加锁的情况下进行，因为读操作不会修改数据，因此它们可以同时发生。读操作在读取数据之前会对其进行引用计数
2. 拷贝 copy：当需要进行写操作时，RCU并不直接在原始数据上进行修改，而是创建一个数据的副本。这样，读操作可以继续在原始数据上进行，不受影响
3. 更新 update：更新操作在新的副本上完成，而不是直接在原始数据上进行修改。一旦更新完成以及在所有进行读访问的使用者结束对旧副本的读取之后，**指针将指向新的数据**，这个过程是原子的。对于已经在进行的读操作，它们会继续引用原始数据，直到它们完成为止。这确保了读操作不会在修改过程中受到干扰
4. 回收 reclaim：一旦没有任何读操作引用原始数据，该数据可以被安全地回收

# 中断 & 异常

* 上半部 top half / 硬中断 hardirq：用来**快速处理**中断，它在中断禁止模式下运行，主要处理跟硬件紧密相关的或时间敏感的工作
* 下半部 bottom half / 软中断 softirq：**延迟处理**上半部未完成的工作，通常以内核线程的方式之后运行





每个 CPU 都对应一个软中断内核线程，名字为 `ksoftirqd/CPU ID`，比如说， 0 号 CPU 对应的软中断内核线程的名字就是 `ksoftirqd/0`

```cmd
$ ps aux | grep softirq
root          13  0.0  0.0      0     0 ?        S    Jan18   0:12 [ksoftirqd/0]
root          22  0.0  0.0      0     0 ?        S    Jan18   0:12 [ksoftirqd/1]
```

这两个线程的名字外面都有中括号，这说明 ps 无法获取它们的命令行参数（cmline）。一般来说，如果 ps 的输出中，名字括在中括号里的，一般都是内核线程







软中断不只包括了刚刚所讲的硬件设备中断处理程序的下半部，一些内核自定义的事件也属于软中断，比如内核调度和 RCU 锁, Read-Copy Update 等

## *proc*

### /proc/softirqs

/proc/softirqs 提供了软中断的运行情况，即各种类型软中断在不同 CPU 上的累积运行次数

```cmd
$ cat /proc/softirqs
                    CPU0       CPU1
          HI:          1          0
       TIMER:   13092424   30698060
      NET_TX:          2          0
      NET_RX:    5081382    5097039
       BLOCK:     428631    5267739
    IRQ_POLL:          0          0
     TASKLET:      11144       9129
       SCHED:   33957966   48773573
     HRTIMER:         43         64
         RCU:   29092449   29456840
```

* HI 高优先级软中断：这是处理紧急事件的软中断，例如处理网络数据包的高优先级任务
* TIMER 定时器软中断：用于处理定时器事件，例如调度进程
* NET_TX 网络传输软中断：处理网络数据包的传输，从内核到网络接口
* NET_RX 网络接收软中断：处理接收到的网络数据包，从网络接口到内核
* BLOCK 块设备软中断：用于块设备的I/O操作，例如磁盘读写
* IRQ_POLL 中断轮询软中断：用于轮询中断状态，以检测中断事件
* TASKLET 任务软中断：用于延迟执行一些任务，通常在中断上下文中
* SCHED 调度软中断：用于系统调度，例如任务的切换和调度
* HRTIMER 高精度定时器软中断：用于处理高分辨率定时器事件
* RCU 读-复制更新软中断：用于在不阻塞读取的情况下执行更新操作

### /proc/interrupts

/proc/interrupts 提供了硬中断的运行情况

# 进程调度

http://www.wowotech.net/process_management/447.html

## *核心数据结构*

### 公共部分

Linux将调度器公共的部分抽象出来，使用`struct sched_class`结构体描述一个具体的调度类

```c
struct sched_class {
	const struct sched_class *next;
	void (*enqueue_task) (struct rq *rq, struct task_struct *p, int flags);
	void (*dequeue_task) (struct rq *rq, struct task_struct *p, int flags);
	void (*check_preempt_curr)(struct rq *rq, struct task_struct *p, int flags);
	struct task_struct * (*pick_next_task)(struct rq *rq, struct task_struct *prev, struct rq_flags *rf);
    /* ... */
}; 
```

1. next：指向下一个调度类（比自己低一个优先级）。在Linux中，每一个调度类都是有明确的优先级关系，高优先级调度类管理的进程会优先获得cpu使用权
2. enqueue_task：入队列，向该调度器管理的runqueue中添加一个进程
3. dequeue_task：出队列，向该调度器管理的runqueue中删除一个进程
4. check_preempt_curr：当一个进程被唤醒或者创建的时候，需要检查当前进程是否可以抢占当前cpu上正在运行的进程，如果可以抢占需要标记TIF_NEED_RESCHED flag
5. pick_next_task：从runqueue中选择一个最适合运行的task。问题是我们依据什么标准来挑选最适合运行的进程呢？一般是按照优先级的顺序

### Linux调度器

Linux支持的调度器有

* RT scheduler 实时进程的实时调度器：用 `rt_sched_class` 来描述，调度策略有SCHED_FIFO 和 SCHED_RR
* CFS scheduler 普通进程的完全公平调度器：用 `rt_sched_class` 来描述，调度策略有SCHED_NORMAL和SCHED_BATCH
* Deadline scheduler：用 `dl_sched_class` 来描述，调度策略为SCHED_DEADLINE
* Idle scheduler 空闲调度器：用 `idle_sched_class` 来描述，调度策略有为SCHED_IDLE

### Deadline & Idle scheduler

这两种调度方式不太常用，所以这简单介绍一下

* Deadline Scheduler截止期限调度器
  * 原理：Deadline 调度器是一种实时 I/O 调度器，旨在确保 I/O 请求在给定的截止期限内完成。它通过为每个请求设置截止期限，优先执行截止期限最近的请求。这有助于避免 I/O 请求因等待时间过长而导致性能下降
  * 特点：
    * 通过 I/O 请求的截止期限进行调度
    * 对于实时任务，可以提供更可预测的 I/O 响应时间
  * 适用场景：适用于对 I/O 响应时间要求敏感的实时系统
* Idle Scheduler空闲调度器
  * 原理：空闲调度器主要用于在系统处于空闲状态时执行 I/O 操作。当系统没有其他任务运行时，空闲调度器可以执行挂起的 I/O 操作，以充分利用系统资源
  * 特点：
    * 在系统空闲时执行 I/O 操作，以提高资源利用率
    * 不会影响其他活动任务的性能
  * 适用场景：适用于需要在系统空闲时执行 I/O 操作的场景，以防止浪费系统资源

## *普通进程调度*

Completely Fair Scheduler, CFS 完全公平调度器

### vruntime

```c
static const int prio_to_weight[40] = {
    /* -20 */     88761,     71755,     56483,     46273,     36291,
    /* -15 */     29154,     23254,     18705,     14949,     11916,
    /* -10 */      9548,      7620,      6100,      4904,      3906,
    /*  -5 */      3121,      2501,      1991,      1586,      1277,
    /*   0 */      1024,       820,       655,       526,       423,
    /*   5 */       335,       272,       215,       172,       137,
    /*  10 */       110,        87,        70,        56,        45,
    /*  15 */        36,        29,        23,        18,        15
};
```

若新进程的vruntime初始值为0，会导致新进程立刻被调度，而且在一段时间内都是最小的

### runqueue

系统中每个CPU都会有一个全局的就绪队列 cpu runqueue，使用`struct rq`结构体描述，它是per-cpu类型，即每个cpu上都会有一个`struct rq` 结构体

<img src="vruntime红黑树.png">

## *实时进程调度*

### SCHED_RR 策略

SCHED_RR 采用 round robin 的策略，比 SCHED_FIFO多维护了一个时间片，相同优先级之间的进程能够轮流执行，每次执行的实际是一个固定的时间片

### SCHED_FIFO 策略

SCHED_FIFO 是一种 run to completion 的算法，采用先进先出的策略，获得CPU 控制权的进程会一直执行直到主动放弃CPU或者被更高优先级的实时进程抢占

## *实验*

Linux的调度器为非实时的进程预留了5%的CPU时间片，避免某死循环实时进程完全占满了CPU

```cmd
$ sysctl -a | grep sched_rt_

kernel.sched_rt_period_us = 1000000
kernel.sched_rt_runtime_us = 950000 # 在period 时间里RT进 程最多能运行的时间
```

`busy -j2` 这个程序调度进程变成实时调度的时候，CPU占用率会从200%变成190%，因为一个线程会留出5%给普通调度的进程



两个都改为FIFO，则同优先级的进程会把另一个实时进程的抢占完，比普通进程还惨









### chrt

chrt 用于为特定的进程或线程设置不同的调度策略和优先级，如实时调度（SCHED_FIFO、SCHED_RR）或普通调度（SCHED_OTHER）

```cmd
$ chrt [options] [priority] command [arguments...]
```

* options 是一些可选的标志，用于指定调度策略等参数
* priority 是进程的优先级，一般是一个整数值，实时调度策略的优先级范围通常是 1 到 99
* command 是要执行的命令
* arguments 是命令的参数

以下是一些常用的选项

* -m 或 --max：显示指定调度策略的最大优先级值
* -p 或 --pid：指定一个已存在的进程的 PID，而不是启动新进程
* -a 或 --all-tasks：operate on all the tasks (threads) for a given pid
* -r 或 --rr：将进程设置为实时轮转调度策略（SCHED_RR）
* -b 或 --batch：将进程设置为批处理调度策略（SCHED_BATCH）
* -f 或 --fifo：将进程设置为实时先进先出调度策略（SCHED_FIFO）

一个常用的组合

# 设备驱动

<img src="Linux驱动结构.drawio.png" width=50%>

## *IDR机制*

IDR (ID Radix) 机制将对象的身份鉴别号整数值ID与对象指针建立字典，从而完成从ID与指针之间的相互转换。IDR机制使用 Radix Tree 作为由id进行索引获取指针的稀疏数组，通过使用位图可以快速分配新的ID，IDR机制避免了使用固定尺寸的数组存放指针。IDR机制的API函数在 `lib/idr.c` 中实现

# 块设备驱动

<img src="Linux块设备文件系统结构.drawio.png" width="50%">

```c
/* 定义在 <fs/block_dev.c> 中 */
struct bdev_inode {
    struct block_device bdev;
    struct inode vfs_inode;
};
```



### bio

```c
struct bio {
	sector_t             bi_sector;           /* 块IO操作的第一个磁盘扇区 */
    struct bio*          bi_next;             /* 链接到请求队列中的下一个bio */
    struct block_device* bi_bdev;             /* 指向块设备描述符的指针 */
    unsigned long        bi_flags;            /* bio的状态标志 */
    unsigned long        bi_rw;               /* IO操作标志 */
    unsigned short       bi_vcnt;             /* bio的 bio_vec 数组中段的数目 */
    unsigned short       bi_idx;              /* bio 的bio_vec 数组中段的当前索引值 */
    unsigned short       bi_phys_segments;    /* 合并之后 bio 中物理段的数目 */
    unsigned short       bi_hw_segments;      /* 合并之后硬件段的数目 */
    unsigned int         bi_size;             /* 需要传送的字节数 */
    unsigned int         bi_hw_front_size;    /* 硬件段合并算法使用 */
    unsigned int         bi_hw_back_size;     /* 硬件段合并算法使用 */
    unsigned int         bi_max_vecs;         /* bio 的bio_vec数组中允许的最大段数 */
    struct bio_vec*      bi_io_vec;           /* 指向 bio 的bio_vec数组中的段的指针 */
    bio_end_io_t*        bi_end_io;           /* bio 的IO操作结束时调用的方法 */
    atomic_t             bi_cnt;              /* bio 的引用计数器 */
    void*                bi_private;          /* 通用块层和块设备驱动程序的IO完成方法使用的指针 */
    bio_destructor_t*    bi_destructor;       /* 释放bio时调用的析构方法（通常是bio-destructor()方法） */
};
```

```c
struct bio_vec {
    struct page* bv_page;      /* 说明指向段的页框中页描述符的指针 */
    unsigned int bv_len;       /* 段的字节长度 */
    unsigned int bv_offset;    /* 页框中段数据的偏移量 */
};
```

# VFS



笔者的理解是 VFS inode 是一切的核心，因为它的 i_ino 在 VFS 和各种不同的文件系统中都是独一无二的。底层的文件系统是通过自己的 inode 来找到具体的物理存储地址的



## *Overview*

```c
/* 定义在 <sched.h> 中 */
struct task_struct {
    // ...
    int link_count, total_link_count; /* 用于在查找环形链表时防止无限循环 */
    struct fs_struct    *fs;          /* 文件系统信息 */
    struct files_struct *files;       /* 打开文件信息 */
    struct nsproxy      *nsproxy;     /* 命名空间 */
    // ...
};
```

不要混淆 `struct fs_struct *fs` 和 `struct files_struct *files`

* 前者保存了进程相关的文件系统的数据，比如说当前工作目录和 chroot 资源限制有关的信息

* 后者则保存的则是当前进程的所使用的各个文件描述符的数据信息

  注意⚠️：这里的文件描述符可不是进行系统编程时的那个 fd 整数，而是一个确切的数据结构。既然我们用到了那个整数，那么 `struct files_struct` 必然是有数据类型可以完成二者之间的映射

VFS这章笔者会分成 VFS 对文件系统的管理和对文件本身的管理两大部分

下图是关于 VFS 中对于文件管理的总体结构

<img src="VFS.drawio.png">

有两条路线可以走

* `super_blcok`
* `task_struct`

## *文件的标识符 inode*

### inode

```c
/* 定义在 <fs.h> 中 */
struct inode {
    struct hlist_node  i_hash; 
    struct list_head   i_list; 
    struct list_head   i_sb_list;
    struct list_head   i_dentry; 
    unsigned long      i_ino;    /* inode 的编号（ID）*/
    atomic_t           i_count;  /* 计数器，记录了访问该inode结构的进程数目 */
    unsigned int       i_nlink;  /* 计数器，记录了使用该inode的硬链接总数 */
    uid_t  i_uid;
    gid_t  i_gid;
    dev_t  i_rdev;
    unsigned long      i_version;
    loff_t             i_size;  /* 文件长度 in Bytes */
    struct timespec    i_atime; /* 最后访问的时间 */
    struct timespec    i_mtime; /* 最后修改的时间 */
    struct timespec    i_ctime; /* 最后修改inode的时间 */
    unsigned int  i_blkbits;
    blkcnt_t      i_blocks;     /* 指定了文件按块计算的长度 */
    umode_t       i_mode;       /* 访问权限 */
    struct inode_operations      *i_op;  /* inode 操作的函数指针集合 */
    const struct file_operations *i_fop; /* 文件操作集合：此前为->i_op->default_file_ops */
    struct super_block           *i_sb;
    struct address_space         *i_mapping;
    struct address_space          i_data;
    struct dquot                 *i_dquot[MAXQUOTAS];
    struct list_head              i_devices;
    union {
        struct pipe_inode_info *i_pipe; /* 用于实现管道的inode的相关信息 */
        struct block_device    *i_bdev; /* 用于块设备 */
        struct cdev            *i_cdev; /* 用于字符设备 */
    };
    int            i_cindex;
    __u32          i_generation;
    unsigned long  i_state;
    unsigned long  dirtied_when; /* 第一个脏操作发生的时间, 以jiffies计算 */
    unsigned int   i_flags;
    atomic_t       i_writecount;
    void          *i_security;
};
```

每个文件（目录也是文件）在文件系统中都有一个唯一的inode，通过该inode可以定位和管理文件的所有相关信息。当文件被创建时，系统会为其分配一个inode，并将文件的元数据信息存储在这个inode中。当文件被访问或修改时，操作系统会根据inode来定位文件的数据块，从而实现文件的读写操作

注意⚠️：底层的文件系统的 inode 数据结构和 VFS 的 inode 数据结构都是不一样的，即使是原生兼容的 Ext\* 文件系统

* 对于同样支持 inode 的底层文件系统，欠缺的部分由内核自身在从底层文件系统读入信息时生成或动态建立
* 对于那些没有 inode 的底层文件系统，必须从其包含的数据中提取出相关信息并拼接出这里给出的形式

### inode 操作 & 文件操作

inode_operations 和 file_operations 是两个包含了很多函数指针的结构体。file_operations 用于操作文件中包含的数据，inode_operations 则负责管理结构性的操作（比方说删除一个文件）和文件相关的元数据

这里先介绍 inode 操作，对于文件本身的操作 file_operations 会在下面介绍

```c
/* 定义在 <fs.h> 中 */
struct inode_operations {
    int (*create) (struct inode *,struct dentry *,int, struct nameidata *); 
    struct dentry * (*lookup) (struct inode *,struct dentry *, struct nameidata *); 
    int (*link) (struct dentry *,struct inode *,struct dentry *); 
    int (*unlink) (struct inode *,struct dentry *); 
    int (*symlink) (struct inode *,struct dentry *,const char *); 
    int (*mkdir) (struct inode *,struct dentry *,int); 
    int (*rmdir) (struct inode *,struct dentry *); 
    int (*mknod) (struct inode *,struct dentry *,int,dev_t); 
    int (*rename) (struct inode *, struct dentry *, struct inode *, struct dentry *); 
    int (*readlink) (struct dentry *, char __user *,int); 
    void * (*follow_link) (struct dentry *, struct nameidata *); 
    void (*put_link) (struct dentry *, struct nameidata *, void *); 
    void (*truncate) (struct inode *); 
    int (*permission) (struct inode *, int, struct nameidata *); 
    int (*setattr) (struct dentry *, struct iattr *); 
    int (*getattr) (struct vfsmount *mnt, struct dentry *, struct kstat *); 
    int (*setxattr) (struct dentry *, const char *,const void *,size_t,int); 
    ssize_t (*getxattr) (struct dentry *, const char *, void *, size_t); 
    ssize_t (*listxattr) (struct dentry *, char *, size_t); 
    int (*removexattr) (struct dentry *, const char *); 
    void (*truncate_range)(struct inode *, loff_t, loff_t);
    long (*fallocate)(struct inode *inode, int mode, loff_t offset, loff_t len);
}
```

## *VFS对文件本身的组织*

```c
/* 定义在 <sched.h> 中 */
struct files_struct {
    atomic_t count;
    struct fdtable *fdt; /* RCU 指针 */
    struct fdtable fdtab;
    int next_fd; /* 下一次打开新文件时使用的文件描述符 */
    struct embedded_fd_set close_on_exec_init;
    struct embedded_fd_set open_fds_init;
    struct file* fd_array[NR_OPEN_DEFAULT];
};

/* 定义在 `<file.h>` 中 */
struct embedded_fd_set {  unsigned long fds_bits[1];  }; /* 一个简单的 bitmap */
```

在 *系统编程.md* 中在上层应用编程时频繁出现的整数文件描述符 fd 其实就是这里 `fd_array` 的下标

`NR_OPEN_DEFAULT` 是每个进程可以打开的最大值，它定义在 `include/linux/sched.h` 中，默认值为 BITS_PER_LONG，32位和64位系统的值分别为32和64

```c
/* 定义在 <file.h> 中 */
struct fdtable {
    unsigned int max_fds;
    struct file ** fd; /* 当前fd_array */
    fd_set *close_on_exec;
    fd_set *open_fds;
    struct rcu_head rcu;
    struct files_struct *free_files;
    struct fdtable *next;
};
```

### 文件描述符 `struct file`

文件描述符保存了文件本身的特征信息

```c
/* 定义在 <fs.h> 中 */
struct file {
    struct list_head   fu_list; /* 文件对象实例链表 */
    struct path        f_path;  /* 包含目录项 */
#define f_dentry f_path.dentry
#define f_vfsmnt f_path.mnt
    const struct file_operations *f_op; /* 文件操作表 */
    atomic_t      f_count;              /* 文件对象的使用计数 */
    unsigned int  f_flags;              /* 当打开文件时所传递的额外的标志位 */
    mode_t        f_mode;               /* 文件的访问模式 */
    loff_t        f_pos;                /* 文件当前的位移量（文件指针）*/
    struct fown_struct    f_owner;      /* 拥有文件的进程的信息（信号 & 异步IO） */
    unsigned int          f_uid,   f_gid;
    struct file_ra_state  f_ra;         /* 预读状态 */
    unsigned long         f_version;    /* 版本号 */
    // ... 
    struct address_space  *f_mapping;   /* 页缓存映射 */
    // ... 
};

/* 定义在 <namei.h> 中 */
struct path {
    struct vfsmount *mnt;
    struct dentry *dentry;
};
```

path 的 `struct dentry` 缓存提供了文件名和 inode 之间的关联，用于快速寻找文件

### 文件操作

```c
/* 定义在 <fs.h> 中 */
struct file_operations {
    struct module *owner;
    loff_t (*llseek) (struct file *, loff_t, int);
    ssize_t (*read) (struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write) (struct file *, const char __user *, size_t, loff_t *);
    ssize_t (*aio_read) (struct kiocb *, const struct iovec *, unsigned long, loff_t);
    ssize_t (*aio_write) (struct kiocb *, const struct iovec *, unsigned long, loff_t);
    int (*readdir) (struct file *, void *, filldir_t);
    unsigned int (*poll) (struct file *, struct poll_table_struct *);
    int (*ioctl) (struct inode *, struct file *, unsigned int, unsigned long);
    long (*unlocked_ioctl) (struct file *, unsigned int, unsigned long);
    long (*compat_ioctl) (struct file *, unsigned int, unsigned long);
    int (*mmap) (struct file *, struct vm_area_struct *);
    int (*open) (struct inode *, struct file *);
    int (*flush) (struct file *, fl_owner_t id);
    int (*release) (struct inode *, struct file *);
    int (*fsync) (struct file *, struct dentry *, int datasync);
    int (*aio_fsync) (struct kiocb *, int datasync);
    int (*fasync) (int, struct file *, int);
    int (*lock) (struct file *, int, struct file_lock *);
    ssize_t (*sendpage) (struct file *, struct page *, int, size_t, loff_t *, int);
    unsigned long (*get_unmapped_area)(struct file *, unsigned long,
                                       unsigned long, unsigned long, unsigned long);
    int (*check_flags)(int);
    int (*dir_notify)(struct file *filp, unsigned long arg);
    int (*flock) (struct file *, int, struct file_lock *);
    ssize_t (*splice_write)(struct pipe_inode_info *, struct file *,
                            loff_t *, size_t, unsigned int);
    ssize_t (*splice_read)(struct file *, loff_t *, 
                           struct pipe_inode_info *, size_t, unsigned int);
};
```

## *文件系统的特征信息*

```c
/* 定义在 <fs_struct.h> 中 */
struct fs_struct {
    atomic_t count;
    int umask;
    struct dentry * root, * pwd, * altroot;
    struct vfsmount * rootmnt, * pwdmnt, * altrootmnt;
};
```

### VFS命名空间

```c
/* 定义在 <nsproxy.h> 中 */
struct nsproxy {
    // ...  
	struct mnt_namespace *mnt_ns;
    // ...  
};

/* 定义在 <mnt_namespace.h> 中 */
struct mnt_namespace {
    atomic_t count;
    struct vfsmount * root;
    struct list_head list;
    // ...
};
```



## *目录项缓存*

Linux使用目录项缓存（简称dentry缓存）来快速访问此前的查找操作的结果

```c
/* 定义在 <dcache.h> 中 */
struct dentry {
    atomic_t d_count;
    unsigned int d_flags;   /* 由d_lock保护 */
    spinlock_t d_lock;    /* 每个dentry的锁 */
    struct inode *d_inode; /* 文件名所属的inode,如果为NULL,则表示不存在的文件名 */
    /* * 接下来的3个字段由__d_lookup处理
    * 将它们放置在这里,使之能够装填到一个缓存行中*/ 
    struct hlist_node d_hash; /* 用于查找的散列表 */
    struct dentry *d_parent;  /* 父目录的dentry实例 */
    struct qstr d_name;
    struct list_head d_lru;   /* LRU链表 */
    union {
        /* 链表元素,用于将当前dentry连接到父目录dentry的d_subdirs链表中 */
        struct list_head d_child;
        struct rcu_head d_rcu;
    } d_u;
    struct list_head d_subdirs;  /* 子目录/文件的目录项链表 */
    struct list_head d_alias; /* 链表元素,用于将dentry连接到inode的i_dentry链表中 */
    unsigned long d_time;  /* 由d_revalidate使用 */
    struct dentry_operations *d_op;
    struct super_block *d_sb;  /* dentry树的根,超级块 */
    void *d_fsdata;   /* 特定于文件系统的数据 */
    int d_mounted;
    unsigned char d_iname[DNAME_INLINE_LEN_MIN]; /* 短文件名存储在这里 */
};
```

### dentry 操作

```c
/* 定义在 <dcache.h> 中 */
struct dentry_operations { 
    int (*d_revalidate)(struct dentry *, struct nameidata *);
    int (*d_hash) (struct dentry *, struct qstr *);
    int (*d_compare) (struct dentry *, struct qstr *, struct qstr *);
    int (*d_delete)(struct dentry *);
    void (*d_release)(struct dentry *);
    void (*d_iput)(struct dentry *, struct inode *);
    char *(*d_dname)(struct dentry *,char *, int);
};
```

## *VFS对文件系统的操作*

### 注册文件系统

### 装载和卸载

### 管理超级块

## *VFS对文件本身的操作*

VFS以各种系统调用的形式提供了用于文件处理的通用 API

# Ext\* 文件系统

<http://chuquan.me/2022/05/01/understand-principle-of-filesystem/#文件操作>

## *Ext\*文件系统*

### Ext\*文件系统介绍

Linux 的标准文件系统是可扩展文件系统 Extended File System，它经历了几代的发展，包括Ext2、Ext3和Ext4。最常见的版本是 Ext3 和Ext4

* Ext2发布于1993年
  * 作为最初的Ext文件系统的改进版本，它引入了许多重要的特性，如大文件支持和长文件名
  * 适用于小型到中型磁盘，被广泛用于Linux分发版
  * 不支持日志功能，这意味着系统崩溃后的恢复时间可能较长
* Ext3发布于2001年
  * 是EXT2的直接升级，主要新增特性是日志功能 Journaling，日志功能可以显著提高系统崩溃后的恢复速度
  * 向后兼容EXT2，可以轻松从EXT2迁移到EXT3而不需要格式化磁盘
  * 支持在线文件系统检查，减少了维护停机时间
* Ext4发布于2008年
  * 当前Linux系统中使用最广泛的文件系统之一
  * 支持更大的单个文件大小和整体文件系统大小
  * 引入了延迟分配（Extents）、更高效的碎片整理和多块分配等改进，提高了性能
  * 增加了对时间戳的纳秒级精度和更大的卷大小上限

Ext4由于其优越的性能和特性，成为当前Linux系统中的首选文件系统。对于需要高度稳定性和日志功能的系统，Ext3仍然是一个可靠的选择。而Ext2虽然较老，但在某些特定场景（如USB驱动器或其他可移动媒体）中仍然有其用武之地

### Ext\*文件系统组成

作为Linux原生的文件系统，Ext\*文件系统和VFS完全一样，会为每个文件分配两个数据结构：**索引节点 index node** 和**目录项 directory entry**，它们主要用来记录文件的元信息和目录层次结构

* 索引节点 ext\*_inode，用来记录文件的元信息，比如 inode 编号、文件大小、访问权限、创建时间、修改时间、数据在磁盘的位置等等。索引节点是文件的**唯一**标识，它们之间一一对应，也同样都会被存储在硬盘中，所以**索引节点同样占用磁盘空间**。每一个inode是一个大小一般为128字节或256字节的空间
* 目录项 ext*_dentry，用来记录文件的名字、**索引节点指针**以及与其他目录项的层级关联关系。多个目录项关联起来，就会形成目录结构，但它与索引节点不同的是，**目录项是由内核维护的一个数据结构，不存放于磁盘，而是缓存在内存**

### Ex2文件系统的物理结构

<img src="Ex2文件系统物理结构.drawio.png" width="80%">

Linux中可以用 `stat` 来查看文件的具体块情况

* 虽然磁盘的基本单位是512 Byte的扇区 sector（这个是由计算机科学家经过试验设计出来的最优方案），但为了提高读写效率，OS和磁盘进行IO的基本单位是一个块 Block 8*512 Byte = 4KB。这么设计的理由主要有两个

  * 512字节太小，可能会导致多次IO，进而导致系统效率降低
  * 为了解耦硬件和软件设计，若将OS的控制与硬件强耦合，则若未来硬件更改了，OS也要进行大幅修改
* Boot Block 启动块：是为了能够使OS正常启动需要的数据，为了系统的安全可能在每个分区里都会存储一份
* inode
* Block group 块组组成

  * Super Block 超级块：整个文件系统的属性信息，每个block group都有一份备份也是为了整个硬盘的安全。**当文件系统挂载时超级块会被加入内存**
  * GDT 块组描述符：这是当前块组的属性信息，即块组的大小、使用情况、inode使用情况等
  * 用位图进行空闲空间管理 Free space management
    * Block Bitmap：每个bit标识一个Block是否空闲可用
    * inode Bitmap inode位图：每个bit标识一个inode是否空闲可用
  * inode Table 节点表：该块组内所有文件的inode空间的集合
  * Data Blocks：多个4KB块的集合。Linux文件=内容+属性，属性分开保存，因此 Data Blocks 里保存的都是特定文件的内容

## *Ex2的inode*

### inode的设计

inode是文件系统用来管理在磁盘上的文件所用的数据结构，每一个inode都由一个inumber数字隐式引用

* 多级索引 Multi-Level index

  <img src="多级inode设计.drawio.png" width="80%">

  设计inode的时候最重要的问题之一是如何表示一个文件用了哪些block，由于inode本身的大小限制，不可能用任意多个直接指针 `block` 来指明所有的block地址。为了支持大文件，采用了多级索引 Multi-Level index 的设计，用一个间接指针 indirect pointer 来指向其他使用的数据块，被指向的数据块里的数据不是普通的文件数据，而是用来只想其他数据块的指针

  举例来说，一个inode中有12个直接指针，1个间接指针。间接指针指向的一个4KB的Data block有 $4KB/4Byte=1024$ 个指针，因此可以将inode映射的文件的大小扩展为 $(12+1024)\times4KB=4144KB$

  若还是不够，则通过继续套娃得到双重间接指针 double indirect pointer 来支持约4GB大小的文件，甚至是继续的三级间接指针 triple indirect pointer

* 基于范围 Extent-based：一个起始指针+一个存储范围。虽然相比于多级索引减少了inode相关元数据，但不够灵活，不能保证在内存中可以找到一段空余空间

* 基于链接 Linked-based：采用链表的方式将数据块串起来，但存在找尾、不支持随机访问等问题。因此将这些链接保存成一个表，称为 File Allocation Table FAT 文件分配表，用于早期Windows

### Ex2的inode结构

注意区分Ex2文件系统的inode和VFS的inode

```c
/* 定义在 <ext2_fs.h> 中 */
struct ext2_inode {
    __le16 i_mode;  /* 文件模式 */
    __le16 i_uid;  /* 所有者UID的低16位 */
    __le32 i_size;  /* 长度,按字节计算 */
    __le32 i_atime;  /* 访问时间 */
    __le32 i_ctime;  /* 创建时间 */
    __le32 i_mtime;  /* 修改时间 */ 
    __le32 i_dtime;  /* 删除时间 */ 
    __le16 i_gid;  /* 组ID的低16位 */ 
    __le16 i_links_count;  /* 链接计数 */ 
    __le32 i_blocks;  /* 块数目 */ 
    __le32 i_flags;  /* 文件标志 */ 
    union {
        struct {
            __le32 l_i_reserved1;
        } linux1;
        struct {
            // ... 
        } hurd1;
        struct {
            // ... 
        } masix1;
    } osd1;  /* 特定于操作系统的第一个联合 */
    __le32 i_block[EXT2_N_BLOCKS]; /* 块指针(块号) */ 
    __le32 i_generation; /* 文件版本,用于NFS */ 
    __le32 i_file_acl; /* 文件ACL */
    __le32 i_dir_acl; /* 目录ACL */
    __le32 i_faddr;  /* 碎片地址*/
    union {
        struct {
            __u8  l_i_frag;  /* 碎片编号 */
            __u8  l_i_fsize;  /* 碎片长度 */
            __u16  i_pad1;
            __le16  l_i_uid_high;  /* 这两个字段 */
            __le16  l_i_gid_high;  /* 此前定义为reserved2[0] */
            __u32  l_i_reserved2;
        } linux2;
        struct { 
            // ... 
        } hurd2;
        struct {
            // ... 
        } masix2;
    } osd2;   /* 特定于操作系统的第二个联合 */
};
```

### inode与文件名的关系

* Linux中找到文件的过程：inode编号 -> 分区特定的block group -> inode -> 文件属性、内容
* Linux中的inode属性里面，没有保存文件名
* Linux下一切皆文件，因此目录也有自己的inode和data block，目录下的文件的文件名和inode编号映射关系都存储在目录的data block里。文件名和inode编号互为key值

# Page Cache & Buffer Cache



## *Page Cache*

Page 页是内存管理的基本单位，而Page Cache 页高速缓存是操作系统中的一种**内存缓存机制**，用于提高文件系统的性能。它是在文件系统和物理磁盘之间的一层[缓冲区](#VFS支持多文件系统)，将磁盘上的数据以页面的整数倍形式缓存在内存中，以加快对文件的访问速度

当应用程序需要读取文件时，操作系统会首先检查 Page Cache 中是否已经存在所需的页面。如果页面已经缓存在内存中，操作系统可以直接从缓存中读取数据，避免了频繁的磁盘访问。如果页面不在缓存中，操作系统会将该页面从磁盘读取到缓存中，并提供给应用程序使用。类似地，当应用程序写入文件时，操作系统会首先将数据写入 Page Cache，然后在适当的时机再将数据刷写回磁盘

Page Cache 的存在极大地提高了文件系统的读取性能，因为内存的访问速度比磁盘快得多。大多数文件读取操作可以直接在内存中完成，避免了昂贵的磁盘 I/O 操作。此外，Page Cache 还可以提供一定程度的文件写入性能提升，因为数据可以先暂存在内存中，减少对磁盘的频繁写入操作

Page Cache 是透明的，对应用程序而言是不可见的，它是操作系统的一部分。操作系统会根据需要自动管理 Page Cache 的内容，并在内存紧张时自动释放一部分缓存空间

需要注意的是，Page Cache 的存在会导致文件系统的修改对应用程序的可见性延迟。因为数据首先写入到缓存中，而不是直接写入磁盘，所以在数据被刷新回磁盘之前，对文件的修改对其他应用程序是不可见的。这就意味着，如果系统崩溃或断电，尚未刷新到磁盘的数据可能会丢失。为了确保数据的持久性，应用程序可能需要使用类似于 `fsync()` 的操作来强制将数据刷新到磁盘上

Linux系统可以通过读取 `/proc/meminfo` 文件来查看系统内存

Page Cache中的数据是采用基数树组织的，效率很高

### Page & Page Cache

**并不是所有 page 都被组织为 Page Cache**，Linux 系统上供用户可访问的内存分为两个类型

* File-backed pages：文件备份页也就是 Page Cache 中的 page，对应于磁盘上的若干数据块；对于这些页最大的问题是脏页回盘。这种page的swap效率较高
* Anonymous pages：匿名页不对应磁盘上的任何磁盘数据块，它们是进程的运行时内存空间（例如方法栈、局部变量表等属性）。swap效率较低

还要注意：不是所有的磁盘数据都会被加载为内存的Page Cache中，比如Directed IO直接访问了 Buffer cache

[Buffer Cache](#VFS)（缓冲区缓存）是操作系统中另一种用于提高文件系统性能的内存缓存机制。它类似于 Page Cache，但其重点是缓存文件系统层面的数据块，而不是整个文件页面

Buffer Cache 的主要作用是减少磁盘 IO 操作，通过将文件系统的数据块缓存在内存中，以便更快地访问和操作这些数据。当应用程序读取或写入文件系统的数据块时，操作系统会首先检查 Buffer Cache 中是否已经存在所需的数据块。如果数据块已经缓存在缓冲区中，操作系统可以直接从缓存中读取或写入数据，避免了对磁盘的实际读写操作

Linux Kernel 2.4之前，Page Cache 与 buffer cache 是完全分离的。2.4 版本内核之后，两块缓存近似融合在了一起：如果一个文件的页加载到了 Page Cache，那么同时 buffer cache 只需要维护块指向页的指针就可以了。只有Directed IO才会真正放到buffer cache。因此现在说的Page Cache一般就是指这两者的结合

### Page Cache 与文件持久化的一致性&可靠性

任何系统引入缓存，就会引发一致性问题：即内存中的数据与磁盘中的数据不一致。例如常见后端架构中的 Redis 缓存与 MySQL 数据库就存在一致性问题

Linux提供了两种保证一致性的策略：写回与写透

* 写回策略 write-back 是指在更新缓存中的数据时，**只更新缓存而不立即写回到底层存储介质**。当数据被修改后，只会更新到缓存中的对应位置，而不是立即写入到主存或磁盘中。当数据被置换出缓存或缓存行被替换时，才会将被修改的数据写回到底层存储介质。**OS中存在周期性的线程执行写会任务，这是OS默认的方案**
* 写透策略 write-through 是指**在更新缓存中的数据时，同时将数据写回到底层存储介质**。当数据被修改后，会立即更新到缓存中的对应位置，并且同时将修改后的数据写入到主存或磁盘中。向用户层提供特定接口，用户可以自行调用强制写透

写回和写透之间的主要区别在于数据的更新时机和写入的位置：

* 写回 WB：数据被修改后，只更新到缓存中，然后在合适的时机将被修改的数据写回到底层存储介质
* 写透 WT：数据被修改后，立即更新到缓存中，并且同时写入到底层存储介质，保证数据的一致性和持久性

写回策略通常会在缓存的一致性机制中使用，例如在处理器高速缓存（CPU Cache）或磁盘缓存（Disk Cache）中。它可以提高写操作的性能，并减少对底层存储介质的频繁写入。然而，写回策略可能会导致数据的延迟一致性，即在写回之前，其他组件可能无法看到最新的修改。这就需要额外的机制（如写入无效ating或写入回调）来确保数据的一致性

写透策略则更注重数据的持久性和一致性，确保每次写操作都能够立即写入到底层存储介质。尽管写透策略提供了较高的数据一致性，但在写入操作频繁时可能会降低系统的整体性能，因为每次写操作都需要写入到底层存储介质

选择使用写回还是写透策略取决于具体应用场景和需求。写回策略通常用于读写比较频繁且对一致性要求不那么严格的情况，而写透策略则适用于对数据一致性和持久性要求较高的场景

### 写回 & 写透用到的系统调用

文件 = 数据 + 元数据，所以文件的一致性实际上包括两个方面，即数据的一致性和元数据的一致性

根据对数据和元数据持久化的不同表现可以区分3种用到的系统调用

* `fsync(int fd)`：将 fd 代表的文件的脏数据和脏元数据全部刷新至磁盘中
* `fdatasync(int fd)`：将 fd 代表的文件的脏数据刷新至磁盘，同时对必要的元数据刷新至磁盘中，这里所说的必要的概念是指：对接下来访问文件有关键作用的信息，如文件大小，而文件修改时间等不属于必要信息
* `sync()`：对系统中所有的脏的文件数据和元数据刷新至磁盘中

### Pros and Cons of Page Cache

* Pros
  * 提高访问性能：Page Cache将磁盘上的数据缓存在内存中，使得对文件的读取操作可以直接在内存中完成，避免了频繁的磁盘访问。由于内存的访问速度远远快于磁盘，因此可以显著提高文件的访问性能
  * 减少磁盘 I/O：通过缓存热门或频繁访问的数据页，Page Cache可以减少对磁盘的实际读写操作。它充分利用了内存的高速缓存能力，减少了对慢速磁盘的依赖，从而降低了磁盘 I/O 的开销
* Cons
  * 内存占用：Page Cache需要占用一部分系统内存来存储缓存的数据页。如果缓存的数据量很大，可能会占用较多的内存资源，导致可用内存减少，可能影响其他应用程序的运行
  * 数据一致性延迟：由于Page Cache采用写回策略，数据的更新可能存在一定的延迟。当数据被修改后，只会更新到缓存中，而不是立即写入到主存或磁盘中。这可能导致其他组件无法立即看到最新的修改，需要额外的机制来保证数据的一致性
  * 数据丢失风险：由于Page Cache是在内存中存储数据的，当系统崩溃或断电时，尚未写回到磁盘的数据可能会丢失。虽然操作系统会尽力确保数据的持久性，但在某些情况下，仍然存在数据丢失的风险
* 透明性：Page Cache对应用程序是透明的，对应用层并没有提供很好的管理 API，因此应用程序无需显式地管理缓存。操作系统会自动处理缓存的管理和更新，使得应用程序开发变得更简单。但这种透明性不一定是好事，应用层即使想优化 Page Cache 的使用策略也很难进行。因此像InnoDB就实现了自己的16KB单位的内存管理

## *管理 Page Cache*

### address_space 结构体

在 Page Cache 中缓存了很多的页，用于管理它们的数据结构是 `struct address_space` 结构体，它被定义在 `include/linux/fs.h` 中

```c
/* 定义在 <fs.h> 中 */
struct address_space {
    struct inode  *host;                         /* 所有者:inode,或块设备 */
    struct radix_tree_root  page_tree;           /* 所有页的基数树 */
    unsigned int  i_mmap_writable;               /* VM_SHARED映射的计数 */
    struct prio_tree_root  i_mmap;               /* 私有和共享映射的树 */
    struct list_head  i_mmap_nonlinear;          /* VM_NONLINEAR映射的链表元素 */
    unsigned long  nrpages;                      /* 页的总数 */
    pgoff_t  writeback_index;                    /* 回写由此开始 */
    struct address_space_operations *a_ops;      /* 方法,即地址空间操作 */
    unsigned long flags;                         /* 错误标志位/gfp掩码 */
    struct backing_dev_info  *backing_dev_info;  /* 设备预读 */
    struct list_head   private_list;
    struct address_space  *assoc_mapping;
} __attribute__((aligned(sizeof(long))));
```

* host指向address_space对应文件的inode
* address_space中的page cache之前一直是用radix tree的数据结构组织的，tree_lock是访问这个radix tree的spinlcok（现在已换成xarray）。
* i_mmap是管理address_space所属文件的多个VMAs映射的，用priority search tree的数据结构组织，i_mmap_lock是访问这个priority search tree的spinlcok。
* nrpages是address_space中含有的page frames的总数。
* a_ops是关于page cache如何与磁盘（backing store）交互的一系列operations

# 文件访问

## *读写文件*

## *内存映射*

### mmap的原理

<https://nieyong.github.io/wiki_cpu/mmap详解.html>

<https://www.cnblogs.com/huxiao-tee/p/4660352.html>

<img src="mmap原理.png">

管理虚拟进程空间的mm_struct结构体中有mmap指向 vm_area_struct，用于管理每一个虚拟内存段

mmap内存映射的实现过程，大致可以分为三个阶段

1. **进程启动映射过程，并在虚拟地址空间中为映射创建虚拟映射区域**

   1. 进程在用户空间发起系统调用接口mmap
   2. 在当前进程的虚拟地址空间中，寻找一段空闲的满足要求的连续的虚拟地址
   3. 为此虚拟区分配一个vm_area_struct结构，并对这个结构的各个域进行初始化
   4. 将新建的虚拟区结构 vm_area_struct 插入进程的虚拟地址区域链表或树中

2. **调用内核空间的系统调用函数mmap（不同于用户空间函数），实现文件物理地址和进程虚拟地址的一一映射关系**

   1. 为映射分配了新的虚拟地址区域后，通过待映射的文件指针，在文件描述符表中找到对应的文件描述符，通过文件描述符，链接到内核“已打开文件集”中该文件的文件结构体（struct file），每个文件结构体维护着和这个已打开文件相关各项信息
   2. 通过该文件的文件结构体，链接到file_operations模块，调用内核函数 sys_mmap
   3. sys_mmap 通过虚拟文件系统的inode定位到文件磁盘物理地址
   4. 通过remap_pfn_range函数建立页表，即实现了文件地址和虚拟地址区域的映射关系。此时，这片虚拟地址并没有任何数据关联到主存中

3. **进程发起对这片映射空间的访问，引发缺页异常，实现文件内容到物理内存（主存）的拷贝**

   前两个阶段仅在于创建虚拟区间并完成地址映射，但是并没有将任何文件数据的拷贝至主存。真正的文件读取是当进程发起读或写操作时

   1. 进程的读或写操作访问虚拟地址空间这一段映射地址，通过查询页表，发现这一段地址并不在物理页面上。因为目前只建立了地址映射，真正的硬盘数据还没有拷贝到内存中，因此引发缺页异常
   2. 缺页异常进行一系列判断，确定无非法操作后，内核发起请求调页过程
   3. 调页过程先在交换缓存空间（swap cache）中寻找需要访问的内存页，如果没有则调用nopage函数把所缺的页从磁盘装入到主存中
   4. 之后进程即可对这片主存进行读或者写的操作，如果写操作改变了其内容，一定时间后系统会自动回写脏页面到对应磁盘地址，也即完成了写入到文件的过程

注意：修改过的脏页并不会立即更新回文件中，而是有一段时间的延迟，可以调用 `msync()` 来强制同步, 这样所写的内容就能立即保存到文件里了

# 页回收

页帧回收算法 Page Frame Reclaiming Algorithm, PFRA