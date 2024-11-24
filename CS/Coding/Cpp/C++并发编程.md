# 线程间数据共享

## *线程的基本控制*

### C++ 线程库和 C Posix 的区别

Linux 使用的是 Posix 标准的 pthread 线程库，而 Win 则采用 Windows API 定义的线程库。C++ 对此进行了进一步封装，用户可以在 Linux 和 Win 上使用同一套 C++ 语言级别的线程库

Posix 是面向过程的，然后直接通过pid来操纵线程。C++ 的线程库是面向过程的，通过对象指针来操纵对象

### 构造 thread

构造函数：不允许拷贝构造，但可以移动构造

```cpp
thread() noexcept; // default
template <class Fn, class... Args>explicit thread (Fn&& fn, Args&&... args); 
thread (const thread&) = delete; // 禁止拷贝
thread (thread&& x) noexcept; // 移动构造
```

第一种构造函数 default，构造一个线程对象，没有关联任何线程函数，即没有启动任何线程

第二种构造函数构造一个线程对象，并关联线程函数  `fn`，` args1, args2, ...` 为线程函数的参数

* fn 这个 handler 函数调用可以是函数指针、函数对象、lambda 表达式。另外 fn 是一个万能引用，既可以接收左值，也可以接收右值

* 可变函数列表中的参数是传递给 handler 的

  当创建一个新线程时，不能保证原始参数 `args` 的生命周期会持续到新线程开始执行。因此为了避免潜在的引用悬挂问题，虽然这里明明用了可变参数的万能引用，但构造函数内部将每个参数复制或移动到新的存储位置，在那里它们将被安全地保存直到线程函数开始执行，也就是说都是传值拷贝给 fn 的，所以线程内部对参数的改变无法传递到主线程中

  在技术上，这涉及到在底层实现中创建一个新的元组，并将参数打包进该元组

thread 库的 thread 对象设计类似于 posix linux 下的 pthread 库中创建对象的 `pthread_create()`，其中 `void *start_routine` 这个 handler 就是 fn

```c
#include <pthread.h>
int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void *(*start_routine) (void *), void *arg);
```

比较重要的 API 有：

* 一旦线程启动之后，我们必须决定是要等待直接它结束（通过 `join()`），还是让它独立运行（通过 `detach()`），我们必须二者选其一。如果在 `thread` 对象**销毁的时候**我们还没有做决定，则 `thread` 对象在析构函数出将调用 `std::terminate()` 从而导致我们的进程异常退出

  * `join()`：用于等待线程的结束。调用该函数将阻塞当前线程，直到目标线程执行完成（不过可能目标线程在此处调用之前就已经执行完成了，不过这不要紧）。类似于进程的 wait

  * `detach()`：用于分离线程，让目标线程成为守护线程，使得目标线程的执行可以独立于创建线程的线程继续进行，不过之后也不能和它通信了。分离后的线程将在执行完成后自动释放资源

  * `bool joinable() const noexcept;`：检查 `thread` 对象是否代表一个可以被 join 的线程。如果返回 `true`，则可以对该线程调用 `join` 或 `detach`

    默认构造的 thread 对象、已经被 join 的线程、已经被 detach 的线程、移动后的线程可能都是不能被 join 的

* `void swap(thread& other) noexcept;`：交换两个 `thread` 对象所代表的线程

* `std::native_handle_type native_handle();`：获取一个表示线程实现定义的 "native" 句柄的对象

* `static unsigned int hardware_concurrency() noexcept;`：提供一个提示，表明可以同时运行多少个线程，不受性能损失影响。这通常反映了系统的核心数量

### `std::ref`：线程函数的参数引用问题

当想要创建一个线程并且需要向线程函数传递参数时，通常会遇到一个问题：`std::thread` 的构造函数通过值来传递参数，意味着它会对这些参数进行拷贝。如果希望在线程中直接操作原始变量（即通过引用传递），而不是操作其副本，那么需要使用 `std::ref` 函数

`std::ref` 是位于 `<functional>` 头文件中的一个实用函数模板，它可以将参数包装为一个引用包装器对象，该对象可以被拷贝并保持对原始对象的引用（可以理解为强制以左值引用来传）。这样做允许我们安全地通过引用而非值传递参数给线程

```c++
#include <iostream>
#include <thread>
#include <functional>

void threadFunction(int& x) {
    ++x;
}

int main() {
    int value = 0;

    // 创建线程，使用 std::ref 来传递变量的引用
    std::thread t(threadFunction, std::ref(value));

    // 等待线程结束
    t.join();

    // 输出 value 的值，应该为 1，因为线程函数增加了它
    std::cout << "Value: " << value << std::endl;

    return 0;
}
```

### this_thread 命名空间

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

在 C++ 的`thread`库中，有一个重要的命名空间：`std::this_thread`，它是 std 命名空间下的子空间

`std::this_thread` 命名空间包含了与当前线程相关的函数。这些函数可以用于获取当前线程的标识符、休眠当前线程、查询系统支持的并发线程数等

* `std::thread::id get_id();`

  返回值类型为 id 类型，id 类型实际为 `std::thread` 命名空间下封装的一个类，该类中包含了一个结构体

  ```c
  typedef struct {
      /* thread identifier for Win32 */
      void *_Hnd; /* Win32 HANDLE */
      unsigned int _Id;
  } _Thrd_imp_t;
  ```

* 暂停 / 休眠当前线程，下面两个函数都可以用来阻塞当前线程一段时间，这通常用于等待事件或实现简单的定时操作

  * `void sleep_for(const chrono::duration<Rep, Period>& sleep_duration);`：让当前线程暂停执行指定的时间段。这个时间段是以模板参数 `chrono::duration` 的形式传递
  * `void sleep_until(const chrono::time_point<Clock, Duration>& sleep_time);`：让当前线程暂停执行直到指定的时间点

* `void yield();`：是让当前线程让出自己的时间片给其他线程，避免在确定条件未准备好的情况长期轮询占据时间片

举个例子

```c++
#include <iostream>
#include <thread>
#include <chrono>

void thread_function() {
    // 输出当前线程ID
    std::cout << "Thread ID: " << std::this_thread::get_id() << std::endl;

    // 线程休眠100毫秒
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 再次输出当前线程ID
    std::cout << "Thread ID after sleep: " << std::this_thread::get_id() << std::endl;
}

int main() {
    // 创建并启动线程
    std::thread t(thread_function);
    // 让主线程让出CPU时间片
    std::this_thread::yield();
    // 主线程休眠200毫秒
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // 等待线程结束
    t.join();

    return 0;
}
```

### 只调用一次

 `std::call_once` 是一个同步原语，用于确保某个函数在多线程环境中仅被调用一次。它定义在 `<mutex>` 头文件中，并与 `std::once_flag` 类型一起工作，提供了一种简单有效的方式来执行只需进行一次的初始化或设置

当需要延迟初始化（lazy initialization）某些资源，比如单例模式或者全局状态，并且这些资源的初始化必须在多线程环境中只发生一次时，`std::call_once` 非常有用。使用 `std::call_once` 可以避免在每次访问资源时都检查是否已经初始化完成（比如单例模式的双重锁检查），从而提高效率

```c++
#include <iostream>
#include <thread>
#include <mutex>

// 一次性初始化的标志
std::once_flag flag;

void do_once() {
    std::cout << "Called once" << std::endl;
}

void call_once_thread() {
    // 调用 do_once 函数，但由于 std::once_flag，它只会被执行一次
    std::call_once(flag, do_once);
}

int main() {
    // 创建多个线程，它们都尝试调用 do_once 函数
    std::thread t1(call_once_thread);
    std::thread t2(call_once_thread);
    std::thread t3(call_once_thread);

    // 等待所有线程结束
    t1.join();
    t2.join();
    t3.join();

    return 0;
}
```

## *mutex 库*

### 基本 mutex 的使用

`std::mutex`：最基本的互斥量，该类的对象之间不能拷贝，也不能进行移动

* `void lock();`：锁定互斥锁。如果互斥锁已经被其他线程锁定，则当前线程将阻塞直到能够获得锁
* `bool try_lock();`：尝试锁定互斥量，如果互斥量当前没有被锁定，则立即将其锁定并返回 true；如果已经被其他线程锁定，则立即返回 false，而不会阻塞当前线程
* `void unlock()`：解锁互斥锁

### 扩展锁种类

在 `std::mutex` 的基础功能上，针对下面的三个层面扩展出了不同的锁

* 超时
  * `std::timed_mutex`：提供了带有超时功能的互斥锁
  
    ```c++
    // 尝试在指定的时间段内锁定互斥锁
    bool try_lock_for(const std::chrono::duration<Rep, Period>& rel_time);
    // 尝试在指定的时间点之前锁定互斥锁
    bool try_lock_until(const std::chrono::time_point<Clock, Duration>& abs_time);
    ```
  * `std::recursive_timed_mutex`：在递归中加普通锁可能会引起死锁。递归锁加了一层识别，如果调用者是自己就放开
  * `std::shared_timed_mutex`
  
* 可重入
  * `std::recursive_mutex`：允许同一个线程对互斥量多次上锁（即递归上锁），来获得对互斥量对象的多层所有权。这意味着同一个线程可以多次调用 `lock()`，但也必须调用相应次数的 `unlock()` 才能释放互斥锁
  * `std::recursive_timed_mutex`：支持 `std::recursive_mutex` 和 `std::timed_mutex` 的所有方法
  
* 共享锁  / 读写锁。由 C++17 引入。这种锁我们在数据库中介绍过
  
  对于这类互斥体，实际上是提供了两把锁：一把是共享锁，一把是互斥锁。一旦某个线程获取了互斥锁，任何其他线程都无法再获取互斥锁和共享锁；但是如果有某个线程获取到了共享锁，其他线程无法再获取到互斥锁，但是还有获取到共享锁。这里互斥锁的使用和其他的互斥体接口和功能一样。而共享锁可以同时被多个线程同时获取到（使用共享锁的接口见下面的表格）。共享锁通常用在读者写者模型上
  
  * `std::shared_mutex`
  
    `std::mutex` 中的同名接口对于 `std::shared_mutex` 来说是用于获取独占锁的
  
    ```c++
    void lock_shared();     // 获取共享锁，可能会阻塞调用线程直到成功获取锁
    bool try_lock_shared(); // 尝试获取共享锁，如果立即可用，则返回 true，否则返回 false
    void unlock_shared();   // 解锁共享锁
    ```
  
  * `std::shared_timed_mutex`：支持 `std::shared_mutex` 和 `std::timed_mutex` 的所有方法

## *RAII 锁*

C++ 中的互斥管理降低死锁风险的方式是使用 RAII 锁，即将锁托付给对象的生命周期来避免忘记解锁或抛异常的程序跳转而引起的死锁

* RAII 锁
  * lock_guard：实现严格基于作用域的互斥体所有权包装器
  * unique_lock：实现可移动的互斥体所有权包装器
  * shared_lock：实现可移动的**共享互斥体**所有权封装器
  * scoped_lock：用于多个互斥体的免死锁 RAII 封装器
* 锁定策略
  * defer_lock：类型为 `defer_lock_t`，不获得互斥的所有权
  * try_to_lock：类型为 `try_to_lock_t`，尝试获得互斥的所有权而不阻塞
  * adopt_lock：类型为 `adopt_lock_t`，假设调用方已拥有互斥的所有权

### lock_guard

lock_guard 类似智能指针，采用 RAII 思想，在资源获取和离开作用域分别自动加锁解锁。它不允许手动控制锁的状态（即不能显式解锁或重新锁定），但其简单性使它成为确保作用域内互斥锁安全使用的首选

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

### lock_guard  的模拟实现

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

### unique_lock

unique_lock 也是 RAII 锁，但它的使用更加灵活，除了提供了 `std::lock_guard` 所有的功能，还增加了：

- 手动解锁和再锁定能力：`unlock()` 和 `lock()`

- 可移动性：可以从一个 `std::unique_lock` 对象转移锁所有权到另一个

  lock_guard 锁的持有只能在 lock_guard 对象的作用域范围内，作用域范围之外锁被释放，而 unique_lock 对象支持移动操作，可以将 unique_lock 对象通过函数返回值返回，这样锁就转移到外部 unique_lock 对象中，延长锁的持有时间

- 延迟锁定、尝试锁定和带超时的锁定：结合特定的构造函数参数如 `std::defer_lock`, `std::try_to_lock`, `std::adopt_lock`

  lock_guard 在构造时或者构造前（`std::adopt_lock`）就已经获取互斥锁，并且在作用域内保持获取锁的状态，直到作用域结束；而 unique_lock 在构造时或者构造后（`std::defer_lock`）获取锁，在作用域范围内可以手动获取锁和释放锁，作用域结束时如果已经获取锁则自动释放锁

### shared_lock (14)

`std::shared_lock` 是专门用于管理 `std::shared_mutex` 或 `std::shared_timed_mutex` 的共享锁的 RAII 包装器。与 `std::unique_lock` 类似，`std::shared_lock` 支持延迟锁定、手动解锁、移动语义和超时锁定，但只限于共享模式

```c++
std::shared_mutex sharedMtx;
void reader_function() {
    std::shared_lock<std::shared_mutex> lock(sharedMtx);
    // 在此区域中进行只读访问共享数据...
} // lock 在这里离开作用域并自动解锁 sharedMtx
```

### scoped_lock (17)

`std::scoped_lock` 在构造时自动获取一个或多个互斥锁，并在析构时释放这些互斥锁。`std::scoped_lock` 的优势之一是它能够同时锁定多个互斥锁而不会引起死锁，因为它在内部使用了 `std::lock` 函数来进行死锁避免算法

```c++
std::mutex mtx1;
std::mutex mtx2;

void safe_double_function() {
    std::scoped_lock lock(mtx1, mtx2); // 同时锁定 mtx1 和 mtx2
    // 临界区：操作共享资源...
} // lock 被销毁，mtx1 和 mtx2 都被自动解锁
```

# 并发同步

C++ 对 Linux 中几乎所有的并发同步方式都进行了包装

## *条件变量*

C++ 提供了两种条件变量：

- `std::condition_variable`：只能与最基础的非递归互斥锁 `std::unique_lock<std::mutex>` 搭配使用
- `std::condition_variable_any`：可以与任何满足基本锁定和解锁操作的互斥类型一起使用

`std::condition_variable_any` 更通用，是因为它做了更多的处理，因此也可能会产生额外的开销，涉及其性能、自身的体积或系统资源等。因此应该优先使用 `std::condition_variable`

### API

- `wait()`：线程调用 `wait()` 后会被阻塞，直到另外某个线程调用 `notify_one()` 或者 `notify_all()` 通知；在等待期间，互斥锁将被释放，当线程接收到通知并再次获得锁时继续执行
- `wait_for()`：允许线程等待指定的时间段，在超时或收到通知时返回
- `wait_until()`：允许线程等待直到指定的时间点，在超时或收到通知时返回 0
- `notify_one()`：唤醒至少一个等待 `condition_variable` 的线程
- `notify_all()`：唤醒所有等待 `condition_variable` 的线程

### `std::condition_variable` & `std::unique_lock` 配合使用

`std::condition_variable` 被设计为专门与 `std::unique_lock<std::mutex>` 一起工作

这是因为条件变量在 `wait()` 操作中需要临时释放互斥锁并在条件满足后再次锁定。`std::unique_lock` 具备这种能力，即它可以在 `pthread_cond_wait()` 和类似的函数内部临时释放并在返回时重新获取互斥锁。`std::lock_guard` 不具备这种能力，因为它没有提供解锁和再次锁定的功能

```c++
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void print_id(int id) {
    std::unique_lock<std::mutex> lck(mtx);
    while (!ready) { // Handle spurious wake-ups.
        cv.wait(lck);
    }
    std::cout << "Thread " << id << '\n';
}

void go() {
    std::unique_lock<std::mutex> lck(mtx);
    ready = true;
    cv.notify_all(); // Wake up all threads.
}

int main() {
    std::thread threads[10];
    // spawn 10 threads:
    for (int i = 0; i < 10; ++i)
        threads[i] = std::thread(print_id, i);

    std::cout << "10 threads ready to race...\n";
    go(); // Go!

    for (auto& th : threads) th.join();

    return 0;
}
```

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

## *barrier (20)*

`std::barrier` 对于那些需要分阶段执行的任务以及需要在每个阶段结束后同步的情况非常有用。在 barrier 的每个同步周期内，只允许每个线程唯一一次运行到其所在之处。线程运行到线程卡处就会被阻塞，一直等到同组的线程全都抵达，在那个瞬间，它们会被全部释放。然后这个线程卡可以被重新使用

### API

1. 构造函数：`std::barrier` 的构造函数需要两个参数：线程计数和一个可选的回调函数。线程计数表示需要等待的线程数量，回调函数在所有线程达到临界点时被调用
2. `arrive()`：表示线程到达屏障点，并减少计数器，但不等待其他线程
3. `arrive_and_wait()` ：每个线程调用 `arrive_and_wait()` 来通知达到临界点，并等待其他线程到达。一旦所有线程都调用了 `arrive_and_wait()`，它们就会在临界点上同步，然后继续执行
4. `arrive_and_drop()`：与 `arrive_and_wait()` 类似，但是它不会等待其他线程到达临界点，而是立即返回，减少计数器并永久减少参与的线程总数。这对于不需要等待其他线程的情况很有用

## *latch (20)*

### intro

`std::latch` 是 C++20 标准中引入的一种同步原语，用于线程同步。它允许一个或多个线程等待一组操作的完成，直到所有操作完成之后才能继续执行

在创建 `std::latch` 对象时，在构造函数中指定一个计数值。每当一个线程完成了一个任务，就可以通过调用 `count_down()` 来减少 latch 的计数。当计数值变为零时，所有等待的线程都会被释放，可以继续执行

之所以叫做 latch 的原因就是因为一旦它就绪就会保持封禁的状态，直到计数器置 0，对象被清空

一旦 `std::latch` 的内部计数器降至零，它就不能再重新设置或使用了，这与 `std::barrier` 不同（`std::barrier` 用于循环等待点，并且可以重置）

`std::latch` 的用途包括：

- 协调多个线程的启动时间，确保它们在继续之前都准备好了
- 等待一个由多个并发操作组成的任务完成

### 示例

```c++
#include <latch>
#include <thread>
#include <iostream>

std::latch latch(2); // 创建一个初始计数器为2的 std::latch

void worker(int id) {
    // 做一些准备工作...
    std::cout << "Worker " << id << " is ready" << std::endl;
    latch.count_down(); // 表示该工作者准备完毕，减少计数器
    latch.wait(); // 等待其他工作者也准备好

    // 做实际工作...
    std::cout << "Worker " << id << " is working" << std::endl;
}

int main() {
    std::thread t1(worker, 1);
    std::thread t2(worker, 2);

    t1.join();
    t2.join();

    return 0;
}
```

## *semaphores (20)*

### counting_semaphore

`std::counting_semaphore` 是一个更通用的信号量类型，它维护一个内部计数器表示可用资源的数量。线程可以通过调用 `acquire()` 来获取资源，这将减少计数器的值，如果计数器已经为零，则线程将阻塞直到其他线程释放资源。线程使用完资源后，应该调用 `release()` 来增加计数器的值，以便其他线程可以使用资源

* `std::counting_semaphore` 的构造函数接受一个无符号整数，用来设置初始的计数器值
* `acquire()`：阻塞当前线程，直到计数器大于零，然后减少计数器
* `try_acquire()`：尝试非阻塞性地减少计数器，如果计数器为零则返回 false
* `release()`：增加计数器，可能会唤醒等待中的线程
* `try_acquire_for()` 和 `try_acquire_until()`：在指定时间段或直到某个时间点尝试获取资源，如果在此期间资源不可用，则返回 false

### binary_semaphore

`std::binary_semaphore` 是 `std::counting_semaphore` 的特殊情况，其计数器只能为 0 或 1，类似于 Linux 中的 PV 操作。它通常用于实现互斥锁和条件变量，并且可以用作简单的锁

## *异步 IO*

C++ 的异步 IO 非常像 JavaScript 的异步机制，即 Promise API

future 对象 `<==>` Promise

### future 模板类

`std::future` 是 C++ 标准库中的一个模板类，它提供了一种访问异步操作结果的机制。当启动一个异步操作时（比如通过 `std::async`、`std::packaged_task` 或者 `std::promise`），可以获得一个与之对应的 `std::future` 对象。这个对象在未来某个时间点会持有异步操作的结果

使用 `std::future` 可以实现以下目的：

1. 从异步操作获取结果
2. 等待异步操作完成
3. 查询异步操作的状态

future 的 API 有：

- `get()`：获取与 `std::future` 对象相关联的共享状态的值。如果共享状态包含一个有效结果，`get()` 会返回这个结果；如果共享状态包含一个异常，则会抛出这个异常。**调用 `get()` 会使 `std::future` 对象变为空，即不再包含共享状态**

- `wait()`：等待与 `std::future` 对象相关联的共享状态就绪，也就是等待异步操作完成。如果共享状态已经就绪，`wait()` 会立即返回

- `wait_for()`：等待共享状态就绪一段指定的时间。如果在指定时间内共享状态变为就绪，它返回 `std::future_status::ready`；否则返回 `std::future_status::timeout`

  ```c++
  if (future.wait_for(std::chrono::seconds(1)) == std::future_status::ready) {
      // 结果已经准备好
  }
  ```

- `wait_until()`：等待共享状态直到达到指定的时间点就绪。如果在指定时间点之前共享状态变为就绪，它返回 `std::future_status::ready`；否则返回 `std::future_status::timeout`

  ```c++
  auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(1);
  if(future.wait_until(deadline) == std::future_status::ready) {
      // 结果已经准备好
  }
  ```

- `valid()`: 检查 `std::future` 对象是否持有与异步操作相关联的共享状态。只有当 `std::future` 对象持有共享状态时，调用 `get()` 才是合法的

  ```c++
  if(future.valid()) {
      // 可以安全地调用 get()
  }
  ```

- `share()`：将 `std::future` 转换为 `std::shared_future`，后者可以被多次拷贝，并允许多个线程等待或访问相同的异步结果

  ```c++
  std::shared_future<int> shared_fut = future.share();
  ```

### shared_future

独占 future, unique future `std::future<>` 和共享 future, shared future `std::shared_future<>` 参考了 `std::unique_ptr` 和 `std::shared_ptr` 的设计 

同一事件仅仅允许关联唯一一个 `std::future` 实例，但可以关联多个 `std::shared_future` 实例

若没有关联数据，我们应使用特化的模板 `std::future<void>` 和 `std::shared_future<void>`

future 对象本身不提供同步访问。 若多个线程需访问同一个 future 对象, 必须用互斥或其他同步方式进行保护

### async

`std::async` 函数模板用于启动一个异步任务，这个任务在后台线程中运行，并返回一个 `std::future` 对象，该对象可用于获取异步任务的结果。使用 `std::async` 可以简化多线程代码的编写，因为它抽象了线程的创建和管理过程

```c++
template <class Fn, class... Args> future <typename result_of<Fn(Args...)>::type> async (Fn&& fn, Args&&... args);

template <class Fn, class... Args> future <typename result_of<Fn(Args...)>::type> async (launch policy, Fn&& fn, Args&&... args);
```

`fn` 是指向调用对象的函数指针，`args` 则是传递给调用独享的参数结构体

```c++
#include <future>
#include <iostream>

int performComputation(int value) {
    // 忽略实际的计算过程
    return value * 2;
}

int main() {
    // 使用 std::async 启动异步任务
    auto fut = std::async(performComputation, 10);

    // 执行其他操作...

    // 获取异步任务的结果（如果任务尚未完成，则会阻塞等待）
    int result = fut.get();
    std::cout << "Computed value: " << result << std::endl;

    return 0;
}
```



按默认情况下，`std::async()` 的具体实现会自行决定，即等待 future 时，是启动新线程，还是同步执行任务。大多数情况下，我们正希望如此。但是 C++ 标准还式给给 `std::async()` 补充了一个参数，以指定采用哪种运行方式。参数的类型是 `std::launch`，其值可以是：

* `std::launch::deferred`：指定在当前线程上延后调用任务函数，直到等到在 future 上调用了 `wait()` 或 `get()`，任务函数才会执行。注意：若延后调用任务函数，则任务函数有可能永远不会运行
* `std::launch::async`：指定必须另外开启专属的线程，在其上运行任务函数
* `std::launch::deferred | std::launch::async`，表示由 `std::async()` 的实现自行选择运行方式，这是选项的**默认值**

### 包装任务

`std::packaged_task` 模板类将一个可调用对象包装成一个异步任务，使其能异步执行，并且它的返回值或异常会被存储在与之相关联的 `std::future` 对象中，这使得我们可以手动地控制任务的执行时机，而不需要立即启动一个线程来执行这个任务，如同 `std::async` 所做的

### 异步同步：`std::promise`

`std::promise` 模板类与 `std::future` 紧密配合使用，允许在某个线程中设置一个值或异常，然后在另一个线程中通过关联的 `std::future` 对象来获取这个值或异常。简单来说，`std::promise` 是一个写入端点，而 `std::future` 是一个读取端点

`std::promise` 常常在以下两种情况下使用：

1. 某个具体时间点或事件发生时，需要通知一个或多个等待结果的线程
2. 当任务可能产生异常，并且需要将异常回传给启动任务的线程

下面是 API

- `get_future()`：返回一个 `std::future` 对象，该对象将用于获取与 `std::promise` 对象相关的值

  ```c++
  std::future<int> f = myPromise.get_future();
  ```

- `set_value()`：为 `std::promise` 对象设置一个值，并使得相关的 `std::future` 对象就绪。如果 `set_value` 被成功调用，则之后调用相关 `std::future` 的 `get()` 会返回此值或者不再阻塞（如果已经被阻塞）

  ```c++
  myPromise.set_value(42);
  ```

- `set_exception()`：为 `std::promise` 对象设置一个异常，与 `set_value()` 类似，但是对应的 `std::future` 将会抛出这个异常而不是返回一个值

  ```c++
  try {
      // ...
      throw std::runtime_error("Error occurred");
      // ...
  } catch(...) {
      myPromise.set_exception(std::current_exception());
  }
  ```

- `set_value_at_thread_exit()` 和 `set_exception_at_thread_exit()`：这些函数的作用与 `set_value()` 和 `set_exception()` 类似，但它们不会马上使 `std::future` 就绪，而是延迟到当前线程结束时。这样可以避免在多线程环境下的竞态条件

  ```c++
  myPromise.set_value_at_thread_exit(42);
  ```

# 原子操作

## *无锁编程*

### intro

无锁编程 Lock-Free Programming，是一种并发编程技术，不依赖于传统的锁机制（如互斥锁）来协调线程对共享资源的访问。在无锁编程中，线程尝试不断地执行操作，直到成功为止，而不是在无法访问资源时被阻塞

无锁编程主要依靠原子操作 Atomic Operations 来实现。原子操作是一种不可分割的操作，保证在执行过程中不会被其他线程中断。这就通过我们接下来要描述的 C++ 原子操作库，即 `std::atomic` 类型和相关函数实现，它们可以对基本数据类型进行无锁操作

```cpp
// 纯并行
for (int i = 0; i < n; i++) {
    mtx.lock();
    cout << this_thread::get_id() << ":" << i << endl;
    //std::this_thread::sleep_for(std::chrono::miliseconds(100));
    mtx.unlock();
}

// 纯串行
mtx.lock();
for (int i = 0; i < n; i++) {
    cout << this_thread::get_id() << ":" << i << endl;
    //std::this_thread::sleep_for(std::chrono::miliseconds(100));
}
mtx.unlock();
```

上面两种加锁的粒度不一样，虽然理论上纯并行的效率应该远高于纯串行。但是加锁粒度过小时，线程上下文切分太过频繁，反而会导致效率变低

CPU 硬件直接提供 CAS，从而实现了原子操作。原子操作不需要上锁，但是它保证了在某个线程使用某个变量（或者说某种资源）的时候不会被其他线程干扰（实际上是若多个操作访问临界资源的时候只有一个写入能成功）

虽然当原子操作写入的时候可能需要多次尝试也会有消耗，但是对于非常细粒度的操作和频繁的线程上下文切换比起来还是高效的多

### 优势

* **性能提升**：由于减少了线程阻塞和上下文切换，无锁编程可以显著提高程序的性能，特别是在高并发环境中
* **避免死锁和饥饿**：传统的锁机制可能导致死锁或线程饥饿的问题，无锁编程可以有效地避免这些问题
* **实时系统中的应用**：在要求高响应性的实时系统中，比如智能驾驶的域控制系统，无锁编程由于其较低的延迟特性，被广泛应用

### 无锁数据结构

无锁数据结构的核心思想是借助原子操作来进行同步，具体见下章

## *CAS*

atomic 库封装了 CAS 操作，CSA 操作是用于实现多线程同步的原子操作。这种操作在现代微处理器和并发编程中被广泛使用，CAS是构建各种无锁数据结构和同步原语的核心，例如：

- 无锁栈、队列和其他数据结构
- 实现自旋锁和其他更复杂的锁机制
- 执行计数器更新和其他累加操作

### 模拟实现

CAS, Compare And Swap：假设内存中存在一个变量 `i`，它在内存中对应的值是 A（第一次读取），此时经过计算之后，要把它更新成 B，那么在更新之前会再读取一下 `i` 现在的值 C，若在业务处理的过程中i的值并没有发生变化，即 A 和 C 相同，才会把 `i` 更新/交换为新值 B。如果 A 和 C 不相同，那说明在计算时，`i` 的值发生了变化，则不更新/交换成 B。最后，CPU 会将旧的数值返回。而上述的一系列操作由 CPU 指令来保证是原子的

CAS操作包含三个参数：

1. **目标内存位置** (`ptr`): 指向要更新的变量的指针
2. **期望值** (`expected`): 预期目标位置上的值
3. **新值** (`new_value`): 如果目标位置上的值与预期值相匹配，则将其更新为此值

操作执行的伪代码描述如下：

```c++
template<typename T> bool compare_and_swap(T* ptr, T expected, T new_value) {
    if (*ptr == expected) {
        *ptr = new_value;
        return true;
    } else {
        return false;
    }
}  fdsa fasd
```

实际的硬件实现是原子的即检查值、可能的替换和返回结果的整个过程是不可分割的，不会被其他线程中断。

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

C++11 标准库中的 `<atomic>` 头文件提供了一系列原子操作 API，它们允许程序员在多线程环境下实现锁自由（lock-free）和等待自由（wait-free）的同步。这些 API 是围绕 `std::atomic` 类模板及其特化构建的

### atomic 模板类

原子操作禁止了拷贝、赋值拷贝构造

对于 POD 类型的数据，atomic 类模板有两种使用方式

1. 可以使用 atomic 类模板，定义出需要的任意原子类型，比如 `atomic<int>`
2. `<atomic>` 已经提前为 POD 定义好了特化模板，比如 `atomic_int`

但是对于自定义数据模型，就只能用 `atomic<>` 新特化一个了，但是该类型必须是平凡可复制 TriviallyCopyable（该类型不需要任何非默认的操作来复制其值），并且不能有自定义的析构函数

### 原子操作

程序员不需要对原子类型变量进行加锁解锁操作，线程能够对原子类型变量互斥的访问

- `is_lock_free()`：可以使用它来查询特定 `std::atomic` 对象是否为无锁的
- `load()`：以原子方式读取存储的值
- `store()`：以原子方式写入存储的值
- `exchange()`：以原子方式替换存储的值
- `compare_exchange_weak()` 和 `compare_exchange_strong()`：原子地比较并可能替换存储的值
- `fetch_add()`, `fetch_sub()`, `fetch_and()`, `fetch_or()`, `fetch_xor()`：执行特定的算术或位运算，并返回旧值
- [【C++入门到精通】 原子性操作库(atomic) C++11 [ C++入门 \]_atomic c++-CSDN博客](https://blog.csdn.net/m0_75215937/article/details/135034826)

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

以下是如何使用 `std::atomic_flag` 实现一个简单的自旋锁

```c++
#include <atomic>
#include <thread>
#include <vector>
#include <iostream>

class Spinlock {
public:
    void lock() {
        while (flag.test_and_set(std::memory_order_acquire)) {
            // 自旋等待，直到能够获取锁
        }
    }

    void unlock() {
        flag.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
};

Spinlock spinlock;

void work(int id) {
    spinlock.lock();
    std::cout << "Thread " << id << " acquired the lock." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 模拟工作
    std::cout << "Thread " << id << " released the lock." << std::endl;
    spinlock.unlock();
}

int main() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back(work, i);
    }
    for (auto& thread : threads) {
        thread.join();
    }
}
```

## *同步操作 & 强制次序*

[谈谈 C++ 中的内存顺序 (Memory Order) - Luyu Huang's Blog](https://luyuhuang.tech/2022/06/25/cpp-memory-order.html#memory_order_seq_cst)

[C++之深入探讨同步操作与强制次序 - 冰山奇迹 - 博客园](https://www.cnblogs.com/blizzard8204/p/17536933.html)

### 同步关系

同步关系 synchronizes-with 是指在多线程环境下，一个线程的操作影响到另一个线程的操作。例如，线程 A 向共享变量写入数据，线程 B 从共享变量读取数据，那么线程 A 的写操作与线程 B 的读操作存在同步关系

### 先行关系

先行关系 happens-before 是指在多线程环境下，一个线程的操作在另一个线程的操作之前发生。先行关系可以由程序顺序和同步关系共同决定。如果操作 A 在操作 B 之前发生，那么操作 A 的结果对操作 B 是可见的

严格先行 strongly-happens-before  是先行关系的一个特例，它加强了规则，使某些优化和处理器指令重排成为不可能。如果操作 A 严格先行于操作 B（A strictly happens-before B），则无论运行时的状态如何，A 始终在 B 之前发生，并且在它们之间不存在任何并发

在多数情况下，当谈论程序的顺序性时，我们通常都在讨论严格先行关系

### 内存顺序

内存顺序决定了原子操作的排序限制，C++ 提供了几种内存顺序选项

```c++
enum memory_order {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst,
};
```

- `std::memory_order_relaxed` 松散一致性：最弱的内存顺序模型，除了保证操作的原子性外，不提供任何的操作执行顺序保证。松散一致性可以提高性能，但可能导致线程间的操作顺序不确定
- `std::memory_order_consume` 仅限于依赖于该操作结果的具体操作序列
- `std::memory_order_acquire` 获取一致性：较弱的内存顺序模型，**只保证读操作不被重排序到前面的操作**。获取一致性通常与释放一致性配合使用，实现线程间的同步
- `std::memory_order_release` 释放一致性：较弱的内存顺序模型，**只保证写操作不被重排序到后面的操作**。释放一致性通常与获取一致性配合使用，实现线程间的同步
- `std::memory_order_acq_rel` 获取释放一致性：获取释放一致性是释放一致性和获取一致性的结合，保证写操作不被重排序到后面的操作，且读操作不被重排序到前面的操作
- `std::memory_order_seq_cst` 顺序一致性：最严格的内存顺序模型，要求所有线程看到的操作顺序一致。顺序一致性可以确保线程间的操作按照预期顺序执行，但可能导致性能下降

选择合适的内存顺序可以在保证正确性的同时获得更好的性能

# 无锁数据结构

## *intro*

### Pros & Cons

* Pros
  * **性能**：在高并发环境下，无锁数据结构能提供比锁定机制更好的性能，因为它减少了由于竞争导致的上下文切换和线程调度开销
  * **死锁安全**：由于不使用传统锁，无锁数据结构避免了死锁的可能性
* Cons
  * **复杂性**：实现无锁数据结构通常比使用锁的数据结构要复杂得多，理解和维护难度大
  * **饥饿**：虽然无锁数据结构不会产生死锁，但可能会出现某些线程连续失败重试，导致操作延迟
  * **ABA问题**：这是无锁编程中经典的一个问题，指的是一个位置先后被写入两次相同的值，但中间可能被改变过，这可能导致错误的行为

## *ABA 问题*

ABA 问题是并发编程中的一个知名问题，尤其是在使用无锁数据结构和算法时经常遇到。它描述的是在执行 CAS 操作时可能出现的一种情况：即使共享数据被其他线程改变过，CAS 操作也可能成功，因为最终的值恰好与预期的旧值相同

假设我们有一个共享变量 A，以及两个线程 1 和线程 2：

1. 线程 1 从共享变量读取值 A 并暂停（例如，进行上下文切换或因为其他操作而等待）
2. 线程 2 将共享变量从 A 更改为 B，然后又从 B 更改回 A，之后线程 2 完成执行
3. 线程 1 恢复运行，并进行 CAS 操作，比较共享变量当前值是否仍为 A
4. 因为共享变量的当前值确实为 A（尽管它在此期间发生了更改），所以 CAS 操作会认为没有变化，从而成功地更新了该值

### 为什么是个问题

这看起来可能不是问题，毕竟最终结果和线程 1 的预期一样。然而，如果在值从 A 变为 B 再变回 A 的过程中涉及到的状态对于程序逻辑至关重要，那么这就可能导致严重的错误。例如在无锁链表中，一个节点可能已经被移除并重新插入链表，其内存可能已经被修改或者再次用于其他目的，但是使用 CAS 的线程可能无法察觉到这个变化

### 解决ABA问题的方法

解决 ABA 问题的一种方法是使用版本号或“标记”来跟踪每次变量的修改。在 CAS 操作中，除了检查变量的值之外，还需要检查附加的版本号是否匹配

以下是一种带有版本号的CAS操作的伪代码表示：

```c++
bool compare_and_swap(Node* expectedNode, Node* newNode) {
    // 检查指针值和版本号是否都匹配
    if (ptr->value == expectedNode->value && 
        ptr->version == expectedNode->version) {
        newNode->version = ptr->version + 1; // 更新版本号
        ptr = newNode;
        return true;
    }
    return false;
}
```

在这个示例中，每个节点不仅包含值，还包含一个版本号。任何修改都会增加版本号，这样即使值回到原始状态，版本号也会告诉该值已经被修改过

一些现代架构提供了双字（double-word）原子操作，如“比较并交换双字”（DCAS或CAS2），可以同时检查两个连续的字（word），其中一个可以用作版本号。这样的原子操作可以直接解决ABA问题，而无需额外的同步机制

在实践中，许多应用程序可能不会受到ABA问题的影响，或者影响有限，因此开发者需要根据具体的应用场景权衡是否需要解决ABA问题







### 无锁队列

实现无锁队列可以采用原子操作：尾插的那一步是原子操作

<[无锁队列的实现 | 酷 壳 - CoolShell](https://coolshell.cn/articles/8239.html#comments)>

# 线程管理

# 协程

C++20 引入了协程 coroutine 的概念，**它是一种支持异步执行的函数或子例程**。协程允许函数的执行在某个点上暂停，将控制权交回给调用者，然后在稍后的时间继续执行，而不阻塞线程

协程是一种轻量级线程，它能够处理非阻塞的、异步的任务，如事件循环、IO 操作等，而不需要创建额外的线程

### 协程的使用

### co_await

### co_yield

### co_return

### promise_type



# EM Cpp 编程建议

本章来自 *Effective Modern C++*

## *条款35：优先选用基于任务而非基于线程的程序设计*

## *条款36：如果异步是必要的，则指定 `std::launch::async`*

## *条款37：使 `std::thread` 类别对象在所有路径都不可联结*

## *条款38：对变化多端的线程句柄析构函数行为保持关注*

## *条款39：考虑针对一次性事件通信使用以void为模板类型实参的期望*

## *条款40：对并发使用 `std::atomic`，对特种内存使用volatile*

volatileness 可以翻译为不稳定、挥发性
