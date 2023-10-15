# 类型推导

## *条款1：理解模板类型推导*

```c++
template<typename T> void f(ParamType param); // 函数模板
f(expr); // 函数调用
```

模板类型T的推导取决于实际传入expr和预设ParamType的类型的不同形式的排列组合的共同作用

### ParamType既非指针也非引用

此时就是按值传递，也就是说无论传入的是什么，param都会是它的一个副本。所以下面的性质是很好理解的，反正函数模板内部得到的都是一个实参拷贝，是不会拷贝它的性质的（原理就是所处区域不一样，一个栈区的拷贝常量区里的内存，不会拷贝它常量区的性质），所以对它做什么都不会影响原来的值

1. **忽略expr的引用性 reference-ness、顶层常量性 top-level constness、volatile性**。注意一个点：右值引用得到的仍然是一个左值引用，所以当expr是一个右值引用时候的，它等同于一个左值引用，所以引用性同样会被忽略
2. 将expr剩余的类别与ParamType的类别再进行匹配

```c++
void f(T param) { std::cout << param << std::endl; }
void f(const T param) { std::cout << param << std::endl; }
```

### ParamType是指针或左值/右值引用，但不是万能引用

1. **忽略expr的引用性**，expr的底层常量性和顶层常量性都会被保留下来（回忆：只有当执行对象拷贝的时候才能忽略expr的顶层常量性，这里不是对象拷贝了）
2. 将expr剩余的类别与ParamType的类别再进行匹配

### ParamType是万能引用 `T&&`

在模板推导时 `T&&` 是一个万能引用。注意：只有 `T&&` 是万能引用，`const T&&` 是一个右值引用

* 若传入的expr是个左值，则T和ParamType都会被推导为左值引用。**这是在模板类型推导中，T被推导为引用类别的唯一场景**。这种情况被称为引用折叠 reference collapse，相当于 `int & + int && -> int &`，相当于三个引用折叠成了一个引用
* 若传入的expr是个右值，运用ParamType是非万能引用的指针或引用时的规则

### 数组和函数指针实参的退化问题

在上面三种情况中，当传入的模板参数T是数组和函数的时候，要考虑其退化情况

退化其实在C语言的函数及传参部分就早已经有过了，其实就是数组/函数名等价于首元素指针。它仍然对模板的类型推导有着重要作用，因此这里再提一下

**在一些语境下，数组和函数会退化成指向其首元素的指针**

* 初始化的情况。注意：对数组名或函数名取地址的指针类型 $\neq$ 名字退化成的指针类型

  ```c++
  int array[5] = { 0, 1, 2, 3, 4 }; // array的数据类型是int[5]
  int *ptr = &array; // 数组名退化为指向其首元素的指针
  int (*ptr2)[5] = &array; // 数组名取地址的类型为int数值指针，不等于数组名的类型，即int指针
  int (&ptr)[5] = array; // 数组引用
  ```

* 数组名作为参数传递时会发生退化，下面三种传递方法是完全等价的

  ```c++
  void foo(int a[100]);
  void foo(int a[5]);
  void foo(int *a);
  ```

  传递数组指针就不等价了

  ```c++
  void fun(int (*a)[5]);
  void fun(int (*a)[100 ]);
  ```

特别注意下面几种情况

* 数组名退化成指针的同时顶层const也退化为底层const

  ```c++
  const char str[] = "hello world";
  f("hello world") // const char str*
  ```

* 函数指针没有底层const，底层const意味着可以修改函数内部的内容了，所以当有底层const的函数指针时，编译器会报错。而函数引用的底层const则会直接被编译器忽略

## *条款2：理解auto类型推导*

auto类型推导和模板类型推导的规则基本上完全一致，它们之间可以建立起一一映射的关系，它们之间也确实存在双向的算法变换

采用auto进行变量声明是哦，类别饰词取代了ParamType，所以同样也是分成三种类型来讨论。并且同样的，数组和函数的退化情况也同样适用于auto

```c++
auto x = 28; // x既非指针也非引用
const auto cx = x; // x既非指针也非引用
const auto &rx = x; // rx是个引用，但不是万能引用
auto &&uref1 = x; // 万能引用，左值，绑定的类型为 int&
auto &&uref2 = cx; // 万能引用，左值，绑定的类型为 const int&
auto &&uref3 = rx; // 引用折叠，右值，绑定的类型为 int&&
```

auto类型推导和模板类型推导的唯一区别在于，当用于auto声明变量的初始化表达式是使用大括号括起时，推导所得的型别就属于 `std::initializer_list`，而模板类型推导则不支持

在这种情况下，若 `{}` 内的类型推导失败（最常见的原因可能是因为元素的数据类型不一致），那么也会导致auto的类型推导失败

注意：下面两种写法auto本来都会推导出 `initializer_list`，但是在C++17推出后，`auto x3{20}` 的auto推导类型不再是 `initializer_list`

```c++
auto x3{20}; // int
auto x4 = {27}; // initializer_list
```

此外，C++14中又引入了一种auto可以用作模板类型推导的方式，需要配合decltype一块使用，具体见下一条条款

## *条款3：理解decltype*

[【C++深陷】之“decltype”-CSDN博客](https://blog.csdn.net/u014609638/article/details/106987131)

decltype (declared type)

### 推导规则

重温一下概念：表达式 expression 是由操作数 operands 和运算符 operators 组成的组合，用来执行特定的计算操作并生成一个结果。表达式可以包括各种数据类型的变量、常量、运算符以及函数调用等。表达式的结果不是左值就是右值。比如说取地址和取引用都是表达式，他们返回左值

推导规则可以分为两大类

* decltype + 变量：所有信息都会被保留，数组与函数名也不会退化
* decltyp + 表达式：表达式返回的不是左值就是右值
  * 左值得到左值引用 `T&`
  * 右值得到该类型 T
  * 将亡值得到 `T&&`
* decltype + 函数/仿函数调用：返回返回值的类型。其实这条就是上面 decltype + 表达式的规则，因为函数调用就是表达式。但是有点容易混淆，所以单独列出来说一下

```c++
int a = 10;
int *aptr = &a;
decltype(*aptr) b1; // *a 表达式返回的是左值引用，结果为int &
decltype(&a) b2; // &a 表达式返回的是地址的右值引用， 结果为int
decltype(std::move(a)) b3; // 将亡值，结果为int &&
```

注意：`int a` 是一个变量，`decltype(a)` 得到的类型是int。若想要用它的表达式属性，可以用 `()` 括起来，`decltype((a))` 返回的是 `int &`

decltype 并不会真的取计算表达式的值，编译器只是会分析表达式并得到类型

```c++
decltype(foo_func(param)) my_var; // foo_func 并没有被执行
```

### 使用场景

* C++11的写法：位置返回，此时auto是一个返回值的置位符

  ```c++
  template <typename Container, typename Index> 
  auto testFun(Container &c, Index i) -> decltype(c[i]) {
      // ... do something
      return c[i]
  }
  ```

* C++14的写法：可以直接写auto，但是此时要注意auto走的是模板推导，当接受的ParamType是引用的时候，输入的引用性会被忽略，所以auto得到的不是一个引用

  ```c++
  template <typename Container, typename Index> 
  auto testFun_error(Container &c, Index i) {
      // ... do something
      return c[i]
  }
  
  template <typename Container, typename Index> 
  delctype(auto) testFun_right(Container &c, Index i) {
      // ... do something
      return c[i]
  }
  ```





```c++
template <typename Container, typename Index> 
delctype(auto) testFun_right(Container &&c, Index i) {
    // ... do something
    return std::forward<Container>(c)[i];
}
```







`std::vector<bool>::reference`



### `decltype(auto)`



等价于C++11的auto占位符写法，意思是保存引用性质，否则引用性会被模板脱掉



C++11中，decltype 的主要用途就在于声明那些返回值型别依赖于形参型别的函数模板

## *条款4：掌握查看类型推导结果的方法*

这一条款主要说了一下用IDE、编译器和typeid 运行时输出来查看类型的方法

核心是typeid和type_info，这部分在 *Cpp基础&11.md* - 多态 - RTTI 中有比较详细的笔记了

# auto

## *条款5：优先使用auto，而非显式类型声明*

auto除了避免程序员书写那些过于冗长的类型之外，还能阻止那些因为手动指定类型带来的潜在错误和性能影响



C++14之后lambda形参也可以使用auto了，这直接就变成了一个模板

lambda表达式的返回值一定要用auto

### 类型的跨平台性

当使用 `std::vector<int> v` 的方法 `v.size()` 的时候，它的返回值是 `std::vector<int>::size_type`，这个类型在不同系统上的大小是不同的，如果我们一直用 size_t 来接受的话可能会造成移植问题，所以用auto来自动推导比较好

### 避免因为类型写错而导致的无用的拷贝

```c++
int a = 10;
//float &b = a; // 编译器报错
const float &b = a;
```

C++中有一个上面这种很怪异的现象，引用的时候一定要类型匹配，否则编译器会报错。但是如果是用const引用就可以了。Primer中给出的解释是创建了临时变量然后隐式转换了

现在考虑下面这个场景，我们想要遍历 `std::unordered_map<std::string, int> m` 这个map，通过编译器我们发现auto的实际推导类型为 `std::pair<const std::string, int>`，这和我们显式给出的 `std::pair<const std::string, int>` 并不相符。根据上面的例子，我们可以认为中间必然是会有拷贝和隐式转换的消耗

```c++
std::unordered_map<std::string, int> m{{"hello", 10}，{"world"，5}, { "heihei", 20}};
for (const std::pair<std::string, int> &p : m) { /*遍历*/ }
for (const auto &p : m){ /*遍历*/ }
```

为了避免这种潜在的因为类型错误而导致的性能开销，应该优先使用auto



## *条款6：auto推导若非己愿，使用显示类型初始化惯用法*

### CRTP

[【编程技术】C++ CRTP & Expression Templates_crtp与expression templates-CSDN博客](https://blog.csdn.net/HaoBBNuanMM/article/details/109740504)

奇异递归模板模式(Curiously Recurring Template Pattern) - 吉良吉影的文章 - 知乎 https://zhuanlan.zhihu.com/p/54945314

奇异递归模板模式 Curiously Recurring Template Pattern CRTP 是C++模板编程时的一种惯用法 idiom，它把派生类作为基类的模板参数。更一般地被称作 F-bound polymorphism。1980年代作为F-bound polymorphism被提出。Jim Coplien于1995年称之为CRTP

编译期多态

```c++
template <typename Derived>
struct Base {
	void name() { (static_cast<Derived *>(this)) ->impl(); };
};
struct D1 : public Base<D1> {
	void impl() { std::cout << "D1: :impl" << std::endl; }
};
struct D2 : public Base<D2> {
	void impl() { std::cout << "D2: :impl" << std::endl; }
};
template <typename Derived>
void func(Base<Derived> derived) {
    derived.name();
}
```

### 表达式模板

表达式模板是CRTP的一种应用

延迟计算表达式，从而可以将表达式传递给函数参数，而不是只能传计算结果

节省表达式中间结果的临时存储空间，减少计算的循环次数

### 代理类

代理类 proxy class 是指以模仿和增强一些类型的行为为目的而存在的类

```c++
class MyArray {
public:
    class MyArraySize {
    public:
        MyArraySize(int size) : theSize(size) {}
        int size() const { return theSize; }
        operator int() const { return theSize; }
    private:
        int theSize;
    };

    MyArray(MyArraySize size) : size_(size), data_(new int[size.size()]) {}
    int operator[](int index) {
        return data_[index];
    }
    ~MyArray { delete int[size.size()]; }
    bool operator==(const MyArray &temp) {
        return data_ == temp.data_;
    }
    MyArraySize size() { return size_; }
private:
    int *data_;
    MyArraySize size_;
};
```

上面的内部类MyArraySize就是一个代理类，它是在模仿int

```c++
class MyArray_ {
public:
    MyArray_(int size) : size_(size), data_(new int[size]) {}
    ~MyArray_() { delete int[size_]; }
private:
    int *data_;
    int size_;
};

void func1(MyArray_ arr) {/**/}
```

如果直接用int会怎么样？上面的MyArray_就是直接用了int。一个很明显的问题就是因为它只吃了一个单参数构造，所以当调用 `func1(10)` 的时候发生参数的隐式转换了

为了禁止隐式转换，这时候要把 `MyArray_` 设置为 explicit 来禁止隐式转换，调用的时候 `func1(MyArray_(10))` 这样就可以le



# 转向现代C＋＋

## *条款7：区别使用 `()` & `[]` 创建对象*

### 多样化的初始化方法

可以用下面的程序在Linux上通过 `-fno-elide-constructors` 关闭编译器的所有优化后得到结果

```c++
class A {
public:
  A(int a) : a_(a) {
    std::cout << "A(int a)" << std::endl;
  }
  A(const A &a) {
    std::cout << "A(const A& a)" << std::endl;
  }
private:
  int a_ = 0;
};
```

1. `A a = 10;` 隐式转换：自动先调转换构造产生一个临时量，然后再掉拷贝构造把临时量拷贝给对象

    ```c++
    /* 打印结果
    A(int a)
    A(const A& a)
    */
    ```

2. `A a(10);`：直接调用一次拷贝

    ```c++
    /* 打印结果
    A(const A& a)
    */
    ```

3. `A a = (10);` 和第一种初始化是一样的

4. `A a{10};`：调一次构造

    ```c++
    /* 打印结果
    A(int a)
    */
    ```

5. `A a = {10}` ：调一次构造，C++11时和第四种构造是一样的，C++14开始不一样了

### `{}` 列表初始化的优势

* `{}` 列表初始化的优势：完美解决下面的问题，即 `{}` 可以一次性接收多个参数、`{}` 只需要一次构造

  * `A a = 10;` 的问题
    * `=` 初始化无论如何只能接受一个参数，若把拷贝构造改成需要两个参数，就不能用它了
    * `=` 初始化需要额外进行一次拷贝

  * `A a(10);` 的问题：被用做函数参数或返回值时还是会执行拷贝

* `{}` 列表初始化有一项新的特性：它禁止内置类型之间进行隐式窄化类别转换 narrowing conversion。所谓的隐式窄化类别转换就是指可能导致数据的精度或范围减小，可能会导致数据丢失或截断的类型转换，比如下面这种

  ```c++
  double x = 5.7;
  int y = x; // 隐式窄化类别转换，将5.7转换为5
  ```

* 列表初始化也大大简化了聚合类的初始化，一个拥有众多类成员的聚合类不需要定义复杂的构造函数就可以直接使用列表初始化进行初始化了

* 列表初始化免疫解析问题 most vexing parse

  对于下面这个声明，C++有两种解释方式：对象参数的创建或者函数类型的声明。也就是说下面 `int (value)` 的括号是没有效果的。C++规定了，任何能够解析为声明的都要解析为声明，而这就会带来副作用

  ```c++
  int i(int (value)); // int i (int value)
  TimerKeeper time_keeper(Timer()); // 本意是调用，传入了一个Timer() 匿名对象
  // TimeKeeper time_keeper(Timer (*)()); 解析成了声明
  ```

  若我们时候 `{}` 列表初始化就不会出现这样的解析错误

### array和 `{}` 的坑

[大括号之谜：C++的列表初始化语法解析_too many initializers for-CSDN博客](https://blog.csdn.net/devcloud/article/details/114523118)

## *条款8：优先考虑使用nullptr而非0和NULL*

我们可以通过auto 自动推导来看看nullptr、0和NULL分别是什么类型

```c++
auto a = 0;
auto b = NULL;
auto c = nullptr;
cout << typeid(a).name() << endl; // int
cout << typeid(b).name() << endl; // Win是int，Linux是long
cout << typeid(c).name() << endl; // std::nullptr_t
```

### 正确调用指针版本的函数重载

nullptr 不会造成0和NULL稍不留意就会遭遇的重载决议问题

```c++
void f(int); // f的三个重载版本
void f(bool);
void f(void*);

f(0); // 调用的是f(int)，而不是 f(void*）
f(NULL); // 可能通不过编译，但一般会调用f(int)。从来不会调用f(void*)
```

Linux中NULL的类型为long，long到int、bool和 `void *` 的转换可能是同样好的，此时编译器会报错

对于没有nullptr可用的C++98程序员而言，指导原则是不要同时重载指针类型和整型

下面是 `std::nullptr_t` 的定义， `std::nullptr_t` 可以隐式转换到所有的裸指针 raw pointer（即非智能指针），包括 `void *`。这就是为什么nullptr可以用来赋值给任意类型指针的原因了

```c++
#include <cstddef>
typedef decltype(nullptr) nullptr_t;
```

### 模板推导时不能混用

```c++
template<typename FuncType,
			typename MuxType,
			typename Ptr Typey>
auto lockAndCall (FuncType func,
				MuxType& mutex,
				PtrType ptr) -> decltype (func(ptr)) {
    MuxGuard g(mutex);
	return func(ptr);
}
```



## *条款9：优先考虑别名声明而非typedef*

C语言和C++98都提供了用typedef给类型起别名，从而简化一些特别长的自定义类型

C++11规定了一种新的方法，称为**别名声明 alias declaration** ，用关键字using来定义类型别名，比如

```c++
using iterator = _list_iterator<T, Ref, Ptr>;
```

但是给指针这种复合类型和常量起类型别名要小心一点，因为可能会产生一些意想不到的后果

using相较于typedef的优势主要是在跟模板相关的时候

* typedef只能给一个实例化的类起别名，比如

  ```c++
  typedef Blob<string> StrBlob;
  ```

  若给要给模板起别名，则必须要在定义的类里面

  ```c++
  template<typename T> class myVector1 {
      typedef std::vector<T> type;
  }
  ```

* C++11标准允许我们为类模板直接定义一个类型别名，比如说下面的代码中，将twin定义为两个成员类型相同的一个模板pair的别名

  ```c++
  template<typename T> using myVector2 = std::vector<T>;
  ```

但真正的好处在于using可以避免使用typenmame来避免二义性

```c++
template<template T> Widge {
	typename myVector1<T>::type myVec1; // 使用了依赖名，要用typename
	myVector2<T> myVec2; // 不需要typename
}
```

## *条款15：情况允许的话尽量使用constexpr*

## *理解特殊成员函数的生成*

# 智能指针

# 右值引用

## *条款23：理解 `std::move` & `std::forward`*

### 类型转换模板

标准库的类型转换 type transformation 模板定义在头文件 `<type_traits>` 中，类型转换也是一种萃取器

<img src="type_traits.png" width="60%">

我们来看其中一个模板 `remove_reference` 是如何实现的，这个模板的作用就是脱去引用类型，得到非引用部分的类型。它是通过多次模板特化得到的

```c++
// 最通用版本
template<class T> struct remove_reference {
    typedef T type;
}
// 部分的特例化模板
template<class T> struct remove_reference<T&> { // 脱去左值引用
    typedef T type;
}
template<class T> struct remove_reference<T&&> { // 脱去右值引用
    typedef T type;
}
```

C++11和C++14使用的方式不太一样

```c++
template<class T> Foo {
	using remove_refrence_t1 = typename std::remove_reference<T>::type; // C+=11
    using remove_refrence_t2 = std::remove_remove_reference<T>; // C++14
}
```

### `std::move`

我们首先来看一种错误的实现

```c++
template<typename T>
T &&myMove(T &&param) {
    return static_cast<T &&>(param);
}

int mmm = 10;
int &&nnn1 = mymove(10); // 通过编译
int &&nnn2 = mymove(mm); // 编译报错
```

* `mymove(10)`：万能引用接收右值输入，万能引用再接收输出，得到一个 `int &&myMove<int>(int &&param)` 的模板
* `mymove(mm)`：万能引用接收左值输入，会发生三次引用折叠，即输入一次，`static_cast` 一次，最后输出再一次，所以实际得到的是一个 `int &myMove<int &>(int &param)`，这不符合右值输出所以报错了

标准库中则大概是下面这么实现的，使用了萃取器 `remove_reference`

```c++
// C++11
template<typename T>
typename remove_reference<T>::type&& move(T &&param) {
    using ReturnType = typename std::remove_reference<T>::type &&; // 类型萃取
    return static_cast<T &&>(param);
}
// C++14
template<typename T>
decltype(auto) move2(T &&param) {
    using ReturnType = std::remove_reference<T> &&;
    return static_cast<ReturnType>(param);
}
```

右值绑定到右值引用上的效果和左值绑定到左值引用上的效果是一摸一样的，并不会发生什么突然的析构

我们可以看到move的作用并不是用了之后马上就把资源转移给右值引用，然后直接把资源回收销毁了。资源的回收销毁仍然是由析构以及OS完成的。move的作用仅仅是将类型强转为 `&&`，因此有些人提议说move应该被叫做rvalue_cast

总结来说，move的作用是强转为 `&&`，而强转为 `&&` 的作用则是告诉编译器：**该对象适合被移动，然后编译器会在条件满足的情况下移动它的资源**

```c++
int a = 3;
int &&b = std::move(a);
a++; // 未定义行为，仅为了实验说明，工程中不要这么做！
cout << a << endl; // 4
b++;
cout << b << endl; // 5
```

上面的实验并不会报错，a在被move了之后还可以被访问，就说明此时a的资源都还在。注意：上面的实验中访问move了之后的资源是一种未定义行为，因为此时我们是无法确定资源是否还在的（当然在上面这种很简单的情况下我们知道资源暂时还没有被移动），所以在工程中不能写出这样的代码

### `std::forward`

万能引用接收参数时，传入左值就一定是一个左值引用，但传入右值也一定会产生左值引用。所以万能引用接收参数实际上丢失了参数到底是左值还是右值的信息

但有些时候我们是需要这种信息的，因为我们可能会需要根据是左值还是右值来自适应。这时候我们就需要用`std::forward<T>` 来保留其引用性质

比方说下面这种情况，不论传入的是左值还是右值，若用的是 `process(param)` 都只会调用左值版本；若用的是 `process(std::move(param))`，则都只会调用右值版本

```c++
void process(const A &lvalArg) { // 左值版本
    std::cout << "deal 1valArg" << std::endl;
}
void process(A &&rva1Arg) { // 右值版本
	std::cout << "deal rvalArg" << std::endl;
}
template<typename T> void logAndProcess(T &&param) {
	// process(param); // 一定调用左值版本
	// process(std::move(param)); // 一定调用右值版本 
	process(std::forward<T>(param)); // 实参用右值初始化时，转换为一个右值
}
```

**`std::forward<T>` 的本质是有条件的move，只有当模板参数T用右值初始化时才转换为右值**，而 `std::move` 的本质就是无论左值还是右值统统转为右值

## *条款24：区分万能引用和右值引用*

`T&&` 有两层含义，第一种就是普通的右值引用，它的目标是识别出可移动对象，然后绑定到右值上

第二种含义是万能引用 universal reference，出现在模板和auto这两种类型推导之中

```c++
template<typename T> void f(T&& param); // T是个万能引用
auto&& var2 = var; // var2是个万能引用
```

## *条款25：针对右值引用使用 `std::move`，针对万能引用使用 `std::forward`*

由于万能引用几乎总是要用到转发，因此万能引用也被称为转发引用 forwarding references

## *条款26：*

## *条款27：*

## *条款28：理解引用折叠*

## *条款29：*

## *条款30：熟悉完美转发的失败情形*

# lambda表达式

## *条款31：避免默认捕获模式*

## *条款32：使用初始化捕获将对象移入闭包*

## *条款33*

## *条款34：优先使用lambda，而非 `std::bind`*

# 并发API