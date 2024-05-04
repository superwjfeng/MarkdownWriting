# 线程间数据共享

## *线程的基本控制*

### C++线程库和C Posix的区别

Linux使用的是Posix标准的pthread线程库，而Win则采用Windows API定义的线程库。C++对此进行了进一步封装，用户可以在Linux和Win上使用同一套C++语言级别的线程库

Posix是面向过程的，然后直接通过pid来操纵线程。C++的线程库是面向过程的，通过对象指针来操纵对象

### thread库

构造函数：不允许拷贝构造，但可以移动构造

```cpp
thread() noexcept; // default
template <class Fn, class... Args>explicit thread (Fn&& fn, Args&&... args); 
thread (const thread&) = delete; // 禁止拷贝
thread (thread&& x) noexcept; // 移动构造
```

第一种default，构造一个线程对象，没有关联任何线程函数，即没有启动任何线程

第二种构造一个线程对象，并关联线程函数  `fn`，` args1, args2, ...` 为线程函数的参数

* fn这个handler 函数调用可以是函数指针、函数对象、lambda表达式。注意
* 可变函数列表中的参数都是传值传参的，所以线程内部对参数的改变无法传递到主线程中

```c
#include <pthread.h>
int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void *(*start_routine) (void *), void *arg);
```

回顾一下posix linux下的pthread库中创建对象的 `pthread_create`，其中 `void *start_routine` 这个handler就是fn

比较重要的API有：

* `get_id()`：返回线程的唯一标识符
* `join()`：用于等待线程的结束。调用该函数将阻塞当前线程，直到目标线程执行完成。类似于进程的wait
* `joinable()`
* `detach()`：用于分离线程，使得目标线程的执行可以独立于创建线程的线程继续进行。分离后的线程将在执行完成后自动释放资源

### thread库的命名空间

```cpp
namespace std {
    namespace this_thread {
        // ...
    }
    class thread {
        // ...
    };
    // ...
}
```

在C++的`thread`库中，有一个重要的命名空间：`std::this_thread`，它是std命名空间下的子空间

`std::this_thread`命名空间：`std::this_thread`命名空间包含了与当前线程相关的函数。这些函数可以用于获取当前线程的标识符、休眠当前线程、查询系统支持的并发线程数等

* `get_id()`

  返回值类型为id类型，id类型实际为 `std::thread` 命名空间下封装的一个类，该类中包含了一个结构体

  ```c
  typedef struct {
      /* thread identifier for Win32 */
      void *_Hnd; /* Win32 HANDLE */
      unsigned int _Id;
  } _Thrd_imp_t;
  ```

* `yield` 是让当前线程让出自己的时间片给其他线程，避免在确定条件未准备好的情况长期轮询占据时间片

* `slepp until`

* `slepp for`

### `std::ref`：线程函数的参数引用问题

强制以左值引用来传

## *mutex库*

### mutex的种类

1. `std::mutex`：最基本的互斥量，该类的对象之间不能拷贝，也不能进行移动
2. `std::recursive_mutex`：允许同一个线程对互斥量多次上锁（即递归上锁），来获得对互斥量对象的多层所有权
3. `std::timed_mutex`
4. `std::recursive_timed_mutex`：在递归中加普通锁可能会引起死锁。递归锁加了一层识别，如果调用者是自己就放开

```cpp
for (int i = 0; i < n; i++) {
    mtx.lock();
    try {
        cout << this_thread::get_id() << ":" << i << endl;
        //std::this_thread::sleep_for(std::chrono::miliseconds(100));
    }
    catch (...) { // ... 表示捕获跑出来的任何异常
        mtx.unlock();
        throw;
    }
    mtx.unlock();
}
```

### `std::mutex` 的使用

```c++
void lock();
bool try_lock();
void unlock();
native_handle_type native_handle();
```

* try_lock：尝试锁定互斥量，如果互斥量当前没有被锁定，则立即将其锁定并返回 true；如果已经被其他线程锁定，则立即返回 false，而不会阻塞当前线程
* native_handle：返回互斥量的底层操作系统句柄，以便与操作系统相关的特定功能进行交互

## *RAII 锁*

RAII将锁托付给对象的生命周期来避免抛异常的程序跳转而引起的死锁

### `lock_guard`

lock_guard 类似智能指针，采用 RAII 思想，在资源获取和离开作用域分别自动加锁解锁

````c++
explicit lock_guard (mutex_type& m);          // 禁止隐式类型转换
lock_guard (mutex_type& m, adopt_lock_t tag);
lock_guard (const lock_guard&) = delete;      // 禁止拷贝
````

`std::adopt_lock`：该参数告诉 `std::unique_lock` 或 `std::lock_guard` 构造函数，表示互斥量已经在当前线程上被锁定，不需要再次进行锁定。这意味着当前线程认为自己已经拥有互斥量的所有权，不需要再次对互斥量进行锁定，而是等待它被释放后再解锁

下面是一个使用的例子

```c++
#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx;

void critical_section() {
    std::lock_guard<std::mutex> lock(mtx); // 在作用域内创建 lock_guard 对象，自动锁定互斥锁
    // 在这里执行临界区代码，确保线程安全
    std::cout << "Critical section protected by lock_guard" << std::endl;
    // 当 lock_guard 对象离开作用域时，互斥锁会自动释放
}

int main() {
    std::thread t1(critical_section);
    std::thread t2(critical_section);

    t1.join();
    t2.join();

    return 0;
}
```

也可以加 `{}` 限定，来手动控制局部域进行解锁

```c++
{
    lock_guard<mutex> lk(mtx);
    // 如果存在抛异常就死锁了
    cout << this_thread::get_id() << ":" << i << endl;
}
std::this_thread::sleep_for(std::chrono::milliseconds(100));
```

### lock_guard 的模拟实现

```c++
// RAII
template<class Lock>
class LockGuard {
public:
	LockGuard(Lock &lk)
		:_lock(lk)
	{
		_lock.lock();
		cout << "thread:" << this_thread::get_id() << "加锁" << endl;
	}

	~LockGuard() {
		cout << "thread:" << this_thread::get_id() << "解锁" << endl << endl;
		_lock.unlock();
	}
private:
	Lock &_lock; //锁是不能拷贝的，所以这里用的成员变量是一个引用
};
```

```c++
class Mutex {
 public:
  Mutex() { pthread_mutex_init(&_lock, nullptr); }
  void lock() { pthread_mutex_lock(&_lock); }
  void unlock() { pthread_mutex_unlock(&_lock); }
  ~Mutex() { pthread_mutex_destroy(&_lock); }

 private:
  pthread_mutex_t _lock;
};

// RAII 的加锁风格
class LockGuard {
 public:
  LockGuard(Mutex *mutex) : _mutex(mutex) {
    _mutex->lock();
    std::cout << "加锁成功..." << std::endl;
  }

  ~LockGuard() {
    _mutex->unlock();
    std::cout << "解锁成功...." << std::endl;
  }

 private:
  Mutex *_mutex;
};
```

### `unique_lock`

unique_lock 也是 RAII 锁，但它的使用更加灵活，比如手动锁定和解锁互斥量，以及支持延迟锁定和条件变量等待。在构造 `std::unique_lock` 对象时可以接受额外的参数，即选择是否要锁定互斥量，并且可以随时释放锁或重新锁定互斥量

* lock_guard在构造时或者构造前（`std::adopt_lock`）就已经获取互斥锁，并且在作用域内保持获取锁的状态，直到作用域结束；而unique_lock在构造时或者构造后（`std::defer_lock`）获取锁，在作用域范围内可以手动获取锁和释放锁，作用域结束时如果已经获取锁则自动释放锁。
* lock_guard锁的持有只能在lock_guard对象的作用域范围内，作用域范围之外锁被释放，而unique_lock对象支持移动操作，可以将unique_lock对象通过函数返回值返回，这样锁就转移到外部unique_lock对象中，延长锁的持有时间

## *其他锁*

### 读写锁（17）

C++17 引入了两种新的锁类型 `std::shared_mutex` 和 `std::shared_timed_mutex`

### 递归锁

`std::recursive_mutex`

# 并发同步

## *条件变量*

抢到了锁但不能执行就要释放锁

wait的Pred返回false的时候就阻塞，notify用于唤醒

## *future*

### 等待多个future

`std::experimental::when_all()`

## *时钟*

`std::chrono` 是 C++ 标准库中用于处理时间和时钟的库，头文件是 `<chrono>` 它提供了一种类型安全的方式来处理时间点、时间间隔和时钟。它是 C++11 引入的一个重要部分，用于更精确地处理时间，特别是在多线程和跨平台开发中非常有用

有两种超时 timeout 机制可以选择

* 迟延超时 duration-based timeout：线程根据指定的时长而继续等待，比如 30 毫秒
* 绝对超时 absolute timeout：在某特定时间点 time point 来临之前，线程会一直等待

大部分 wait 函数都具有两种变体，专门处理这两种机制的超时。处理迟延超时的函数变体以 **\_for** 为后缀，而处理绝对超时的函数变体以**\_until** 为后缀

下面是一些 `std::chrono` 的主要用法和重要组件

### 时钟类

 `std::chrono` 提供了三种时钟类，它们的通用结构大概如下

```c++
class GeneralClock {
public:
	// 构造、析构函数等略
    static GeneralClock::time_point now() { return _t; } // 返回当前
    
private:
    time_point _t;
    std::ratio<fraction> _period;
    static bool is_steady;
}
```

* 通过静态成员函数 now 获取某时钟类的当前时刻，它的返回类型是 `some_clock::time_point`
* 时钟类的计时单元属于名为 period 的成员类型，它表示为秒的分数形式。比如说若时钟每秒计数 25 次，那么它的计时单元即为 `std::ratio<1,25>`，即 $\frac{1}{25}$ 秒计时一次
* 若时钟的计时速率恒定（无论速率是否与计时单元相符）且无法调整，则称之为恒稳时钟。恒稳时钟的静态成员函数 `is_steady = true`

下面介绍一下三种时钟类的作用

* `std::chrono::system_clock`：system_clock 表示操作系统的系统时钟。它通常用于获取当前系统的时间，以及与日历时间相关的操作。可以使用`now()`函数来获取当前时间点

  ```c++
  #include <iostream>
  #include <chrono>
  
  int main() {
      std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
      std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
  
      std::cout << "Current time: " << std::ctime(&currentTime) << std::endl;
      return 0;
  }
  ```

* `std::chrono::steady_clock`： `steady_clock`表示一个恒稳时钟，适用于性能测量和计时。它不受系统时间的影响，通常用于测量代码执行时间

  ```c++
  #include <iostream>
  #include <chrono>
  
  int main() {
      std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  
      // 执行一些操作
  
      std::chrono::steady_clock::time_point end = ？std::chrono::steady_clock::now();
      std::chrono::duration<double> duration =
          std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  
      std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
      return 0;
  }
  ```

* `std::chrono::high_resolution_clock`： `high_resolution_clock`是一个高分辨率时钟，通常使用系统提供的最高分辨率计时器（它具备可能实现的最短计时单元）

  `std::chrono::high_resolution_clock` 可能不存在独立定义，而是由 typedef 声明的另一时钟类的别名

  ```C++
  #include <iostream>
  #include <chrono>
  
  int main() {
      std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  
      // 执行一些操作
  
      std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration =
          std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  
      std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
      return 0;
  }
  ```

### 时长类

`std::chrono::duration<>` 类模版表示时间的长度，具有两个模板参数

* 第一个模版参数指明采用何种类型表示计时单元的数量，比如 int、long 或 double
* 第二个参数以不同的时间单位（如秒、毫秒、微秒、纳秒）来表示。`std::chrono` 库也提供了 `std::chrono::seconds`、`std::chrono::milliseconds`、`std::chrono::microseconds`、`std::chrono::nanoseconds` 等用于表示时间单位的类型

### 时间点类

时间点由类模板 `std::chrono::time_point<>` 的实例表示，具有两个模版参数

* 第一个模板参数指明所参考的时钟
* 第二个模板参数指明计时单元 `std::chrono::duration<>` 的特化

时间点类模版支持一些计算

* 可以使用 `+` 和 `-` 运算符来计算时间点之间的时间间隔，或者将时间间隔添加到时间点上
* 时间点的比较：可以使用比较运算符（如 `<`、`<=`、`==`、`!=`、`>=`、`>`）来比较时间点

## *latch计数器*

### intro

`std::latch` 是 C++20 标准中引入的一种同步原语，用于线程同步。它允许一个或多个线程等待一组操作的完成，直到所有操作完成之后才能继续执行

在创建 `std::latch` 对象时，在构造函数中指定一个计数值。每当一个线程完成了一个任务，就可以通过调用 `count_down()` 方法来减少latch的计数。当计数值变为零时，所有等待的线程都会被释放，可以继续执行

之所以叫做 latch 的原因就是因为一旦它就绪就会保持封禁的状态，知道计数器置0，对象被清空

## *barrier*

`std::barrier` 对于那些需要分阶段执行的任务以及需要在每个阶段结束后同步的情况非常有用。在 barrier 的每个同步周期内，只允许每个线程唯一一次运行到其所在之处。线程运行到线程卡处就会被阻塞，一直等到同组的线程全都抵达，在那个瞬间，它们会被全部释放。然后这个线程卡可以被重新使用

1. 构造函数：`std::barrier` 的构造函数需要两个参数：线程计数和一个可选的回调函数。线程计数表示需要等待的线程数量，回调函数在所有线程达到临界点时被调用
2. `arrive_and_wait()` 方法：每个线程调用 `arrive_and_wait()` 方法来通知达到临界点，并等待其他线程到达。一旦所有线程都调用了 `arrive_and_wait()`，它们就会在临界点上同步，然后继续执行。
3. `arrive_and_drop()` 方法：与 `arrive_and_wait()` 类似，但是它不会等待其他线程到达临界点，而是立即返回。这对于不需要等待其他线程的情况很有用



### flex_barrier

# 原子操作

## *CAS*

atomic库封装了CAS操作

CAS, Compare And Swap：假设内存中存在一个变量 `i`，它在内存中对应的值是A（第一次读取），此时经过计算之后，要把它更新成B，那么在更新之前会再读取一下 `i` 现在的值C，若在业务处理的过程中i的值并没有发生变化，即A和C相同，才会把 `i` 更新/交换为新值B。如果A和C不相同，那说明在计算时，`i` 的值发生了变化，则不更新/交换成B。最后，CPU会将旧的数值返回。而上述的一系列操作由CPU指令来保证是原子的

### C++ 的 CAS 方法

https://www.cnblogs.com/gnivor/p/15919850.html

```c++
bool compare_exchange_weak (T& expected, T val, memory_order sync = memory_order_seq_cst) volatile noexcept;
bool compare_exchange_weak (T& expected, T val, memory_order sync = memory_order_seq_cst) noexcept;
bool compare_exchange_weak (T& expected, T val, memory_order success, memory_order failure) volatile noexcept;
bool compare_exchange_weak (T& expected, T val, memory_order success, memory_order failure) noexcept;
```

```c++
bool compare_exchange_strong (T& expected, T val, memory_order sync = memory_order_seq_cst) volatile noexcept;
bool compare_exchange_strong (T& expected, T val, memory_order sync = memory_order_seq_cst) noexcept;
bool compare_exchange_strong (T& expected, T val, memory_order success, memory_order failure) volatile noexcept;
bool compare_exchange_strong (T& expected, T val, memory_order success, memory_order failure) noexcept;
```

## *原子操作类别*

### atomic库

```cpp
//纯并行
for (int i = 0; i < n; i++) {
    mtx.lock();
    cout << this_thread::get_id() << ":" << i << endl;
    //std::this_thread::sleep_for(std::chrono::miliseconds(100));
    mtx.unlock();
}

//纯串行
mtx.lock();
for (int i = 0; i < n; i++) {
    cout << this_thread::get_id() << ":" << i << endl;
    //std::this_thread::sleep_for(std::chrono::miliseconds(100));
}
mtx.unlock();
```

上面两种加锁的粒度不一样，虽然理论上纯并行的效率应该远高于纯串行。但是加锁粒度过小时，线程上下文切分太过频繁，反而会导致效率变低

CPU 硬件直接提供 CAS，从而实现了原子操作。原子操作不需要上锁，但是它保证了在某个线程使用某个变量（或者说某种资源）的时候不会被其他线程干扰（实际上是若多个操作访问临界资源的时候只有一个写入能成功）





https://cplusplus.com/reference/atomic/



原子操作禁止了拷贝、赋值拷贝构造

程序员不需要对原子类型变量进行加锁解锁操作，线程能够对原子类型变量互斥的访问

可以使用atomic类模板，定义出需要的任意原子类型



虽然当原子操作写入的时候可能需要多次尝试也会有消耗，但是对于非常细粒度的操作和频繁的线程上下文切换比起来还是高效的多



### atomic_flag

`std::atomic_flag` 是 C++ 标准库提供的另一种原子操作类型，它是一种非常轻量级的原子布尔类型，基本上就是 `std::atomic<bool>`。通常用于实现低级的自旋锁

atomic_flag 是 lock-free 的，也是库实现中唯一一个保证 lock-free 的

`std::atomic_flag` 只有两种状态：被设置（set）和未设置（clear），因此它不支持像 `std::atomic<int>` 那样的加载和存储操作。相反，`std::atomic_flag` 只提供了两个成员函数来操作它的状态：

```c++
// <atomic>
bool test_and_set (memory_order sync = memory_order_seq_cst) volatile noexcept;
bool test_and_set (memory_order sync = memory_order_seq_cst) noexcept;

void clear (memory_order sync = memory_order_seq_cst) volatile noexcept;
void clear (memory_order sync = memory_order_seq_cst) noexcept;
```





## *同步操作 & 强制次序*

# 无锁数据结构

无锁数据结构的核心思想是借助原子操作来进行同步

实现无锁队列可以采用原子操作：尾插的那一步是原子操作

<[无锁队列的实现 | 酷 壳 - CoolShell](https://coolshell.cn/articles/8239.html#comments)>

# 线程管理

# 协程

C++20 引入了协程 coroutine 的概念，**它是一种支持异步执行的函数或子例程**。协程允许函数的执行在某个点上暂停，将控制权交回给调用者，然后在稍后的时间继续执行，而不阻塞线程

协程是一种轻量级线程，它能够处理非阻塞的、异步的任务，如事件循环、I/O 操作等，而不需要创建额外的线程

### 协程的使用

### co_await

### co_yield

### co_return

### promise_type

