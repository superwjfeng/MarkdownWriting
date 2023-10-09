# 让自己习惯C++

## *条款1：将C++视做一个语言联邦*

### 范式

> A model or example of the environment and methodology in which systems and software are developed and operated. For one operational paradigm there could be several alternative development paradigms. Examples are functional programming, logic programming, semantic data modeling,algebraic computing, numerical computing, object oriented design, prototyping, and natural language dialogue. -- 牛津计算机辞典
>
> 一种环境设计和方法论的模型或范例；系统和软件以此模型来开发和运行。一个现役的范式可能会有数个开发中的替代范式。以下是一些大家比较熟悉的范式：函数化程序设计、逻辑程序设计、语意数据模型、几何计算、数值计算、面向对象设计、原型设计、自然语言。

范式 paradigm 是一组规则、约定和方法，用于定义程序设计语言的结构和特性。它们定义了如何组织和编写代码以解决问题，以及如何使用语言中提供的构造来表示和操作数据。不同的编程语言支持不同的范式，每种范式都具有一组相关的概念和原则

### C++的四种子语言

今天的C++是一门多范式编程语言 Multiparadigm programming language，同时支持过程式 procedural、面向对象 object-oriented、函数式 functional、泛型 generic和元编程 metaprogramming 特性。这种能力和灵活性使C++成为无可比拟的工具，但也会引起一些混乱。这种混乱就是**似乎所有的“规范语法”规则都会有一些例外**

一种理解方法是将C++视做一个语言联邦，在每一种子语言中的语法规则不一定在另一种子语言中完全使用。C++主要有以下几种子语言

<img src="C++联邦.drawio.png" width="50%">

* C++的C语言部分：C++仍然是基于C的。块、语句、预处理器、内置数据类型、数组、指针等都来自于C。在许多情况下，C++提供的解决问题的方法优于C中的对应方法（比如[条款2：尽量以const、enum、inline替换 #define](#条款2)和[条款13：以对象管理资源](#条款13)），但当使用C++的C部分时，高效编程的规则反映了C语言相对有限的功能：没有模板，没有异常，没有重载等
* C++的面向对象部分：C++的这一部分就是C with Classes的全部内容，即类（包括构造函数和析构函数）、封装、继承、多态、虚函数（动态绑定）等。这是C++中面向对象设计的经典规则最直接适用的部分
* C++的模板部分：这是C++的泛型编程部分。事实上模板是如此强大，它们产生了一种全新的编程范式--模板元编程 template metadata programming TMP。[条款48](#条款48)提供了TMP的概述，但除非是一个铁杆模板迷，否则不必过于担心它。TMP的规则很少与主流C++编程交互
* C++的STL部分：STL是一个模板库，但它是一个非常特殊的模板库。它通过一些约定，很好地将容器、迭代器、算法和函数对象地融合在一起。使用STL时需要确保遵循它的约定

### 简单举例

**当在四种不同的子语言之间切换时，为了实现高效的编程需要遵守的规则是不同的**。下面举一个函数传递的例子

* 对于内置类型 built-in type，值传递通常比引用传递更高效
* 但在C++的面向对象部分时，用户定义构造函数和析构函数的存在意味着引用传递到常量通常更好
* 在C++的模板部分尤其如此，因为在那里，你甚至不知道正在处理的对象的类型
* 然而当进入STL时，迭代器和函数对象是基于C中的指针建模的，因此对于STL中的迭代器和函数对象，旧的C值传递规则再次适用

## <span id="条款2">*条款2：尽量以const、enum、inline替换 `#define`*</span>

因为C的历史原因，`#define` 预处理仍然很常用。事实上，C++的发明者 Dr. Bjarne Stroustrup 也致力于去除C++中预处理器的使用

使用 `#define` 预处理替换有两个显著的缺点

1. 不方便追踪错误，编译器不会显式宏名称
2. 即使很小心了，编写宏仍然是极易出错，应该考虑用内联函数来替换。通过使用内联函数模板，可以获得宏的所有效率，以及普通函数的所有可预测行为和类型安全

```c++
#define ASPECT_RATIO 1.653 // 不好
const double aspect_ratio = 1.653; // 用常量来代替
```

下面两种特殊情况需要我们考虑

### 定义常量指针

```c++
#define AUTHORNAME "Scott Meyers"
const char * const author_name = "Scott Meyers"; // const pointer & pointer to a const
const string author_name2 = "Scott Meyers"; // better
```

定义字符串时不是 `const char *`，这只是一个常量指针，但按照 `#define` 或常量的意义应该既是一个指针常量，也是一个常量指针，即 `const char * const`

然而写成 `const char * const` 这种形式很难看，用C++标准库提供的string会好很多

### 类的专属常量

* 静态常量

  要把常量的作用域限制为类，就必须把它设为成员；而要确保最多有一个常量的副本，就必须把它设为静态成员

  然而在C++中，类静态常量的使用非常特殊。一般的静态变量不能给缺省值，只能在类外面给初始值

  但是有例外：`const static int`、`const static char`、`const static bool` 类型的静态变量可以给缺省值，比如哈希桶中的素数size扩容就用到了这个特性

  或者更准确地可以理解为，**上面三种类型的静态常量的初始值如果在类内声明时指定的，那么在类外定义时反而不允许赋初始值**；反过来如果在类外定义指定了初值，那么类内声明就不可以给出

  非常匪夷所思，明明是类内声明却可以给出初始值，但就是这么规定的

  甚至只要不获取它们的地址，那么甚至可以在不提供类外定义的情况下只声明并使用它们

  ```c++
  class GamePlayer {
  public:
      int get_numturns() { return num_turns; }
  private:
      const static int num_turns = 5; // const static 类型的声明，在这里初始化了！
      int scores_[num_turns];
  }
  const int GamePlyaers::num_turns; // const static 类型的定义，只要不获取它们的地址，可以不给出
  ```

* 枚举常量

  另一种方法是使用枚举，反正只要是编译时顺序确定的就行了

  ```c++
  class GamePlayer {
  public:
      int get_numturns() { return NumTurns; }
  private:
      enum { NumTurns = 5 };
      int scores_[NumTurns];
  }
  ```

### 总结

考虑到const、枚举和内联的可用性，对预处理器（特别是 `#define`）的需求减少了，但并没有完全消除。`#include` 仍然是必不可少的，`#ifdef//#ifndef` 继续发挥避免 circular dependency 的重要作用

* 对于简单常量，首选const对象或枚举，而不是 `#define`
* 对于类似函数的宏，优先选择内联函数

## *条款3：尽可能使用const*

### const限定成员函数

这部分在 *Cpp基础&11.md* 中的-类成员中的const成员部分讲过了，可以回顾一下

在成员函数上使用const

* 明确接口意图，告诉用户传入的参数不会被改变
* **保护参数不变性**，使用常量引用可以确保在函数内部不会修改传入的参数值
* **支持常量和非常量**：常量引用可以接受常量和非常量类型的实参。这增加了函数的通用性。这是因为非常量传给const是权限缩小，扩大了传参范围，即const参数和普通参数都可以传给const引用

C++中成员函数可以根据是否可以通过this来修改对象可以分为两类（即是否有const修饰整个成员函数），并且分别重载

**一般返回值是否设置为const &和this是否设置为const都是成对的**。因为不能通过this来修改的时候，同样也不能通过返回的引用来修改

### 位常量性与逻辑常量性

位常量性 bitwise constness 与逻辑常量性 logical constness 是关于类的成员函数在 const 语境下是否能修改成员变量的概念

* 位常量性表示在 const 成员函数中不能修改任何成员变量的值，即不能改变对象的位 bit 表示。这意味着成员变量的所有位都保持不变。C++中的常量性就是逻辑常量性
* 逻辑常量性表示在 const 成员函数中不会修改外部可见的逻辑状态，但可能会修改成员变量的值，只要这些修改是对外部不可见的。逻辑常量性往往依赖于程序员的实现，而不是编译器的强制检查。C++中可以用mutable将非静态数据成员从按位常量的限制中解放出来

### 避免const重载的代码重复

一对重载的const成员函数和非const成员函数可能会有很多内容重复，此时可以考虑复用

问题是谁调用谁？**用非const成员函数来调用const成员函数是安全的**，这样的话可以把修改数据的部分继续添加在非const成员函数中

```c++
class TextBlock {public:
    const char& operator[](std::size_t position) const{
        //...
        //在这种情况下，去掉返回值上的const是安全的，因为调用非const操作符[的人首先必须有一个非const对象。
        //...
        return text[position];
    }
    char& operator[](std::size_t position){ //只需要调用const版本
        //对op[]的返回类型抛弃const //给*this的类型添加const;调用op的const版本[]，否则还是调用自己无限循环
        return const_cast<char&>(static_cast<const TextBlock&>(*this)[position]);
}
```

## *条款4：确定对象被使用前已先被初始化*

C/C++中定义无初值的对象时对象是否会被初始化的规则很复杂。一般来说，如果使用C++的C部分，并且初始化可能会导致运行时成本，则不能保证会初始化。如果使用C++的非C部分。情况则有时会不同。比如说数组不一定保证其内容初始化，而vector的内容必须初始化

最佳方法是始终在使用对象之前初始化它们。对于内置类型的非成员对象，需要手动完成。如

```C++
int x = 0;                                //手动初始化int类型
const char * text = "A C-style string" ;  //指针的手动初始化
double d;                                 //通过读取输入流来“初始化”，定义后确保立刻赋值也是可以的
std: :cin >> d;
```

**对于几乎所有其他的情况下，初始化的责任都落在构造函数身上**

### 构造函数中的初始化顺序

这一部分主要是强调了构造函数中初始化的顺序。这一部分已经在 *Cpp基础&11.md* 中着重论述了，下面给出一些最重要的点，具体的可以回顾 *Cpp基础&11.md* 

* **初始化列表可以认为是成员变量初始化的地方**。初始化列表是**被自动调用**的，即使是完全空的构造函数也会自动调用，因此可以认为构造函数里的是二次赋值，private里的只是声明，真正的初始化是在初始化列表之中。也就是说**实例化的过程是：类属性声明（若有缺省值就直接初始化）`->` 初始化列表赋值 `->` 构造函数内部赋值（若有的话） **
* **成员变量在类中的声明次序就是其在初始化列表中的初始化顺序，与其在初始化列表中的先后次序无关**

### 不同编译单元中定义的非局部静态对象的初始化顺序

单个编译单元是指产生单个目标文件的源代码，比如典型的一对 `.cc` 文件和 `.h` 文件组合

非局部静态对象 non-local static objects：函数内部的静态对象称为局部静态对象（因为它们在函数中是局部的），其他类型的静态对象称为非局部静态对象，比如说外部链接的静态变量（即全局变量）和内部链接的静态变量（非局部静态对象）

现在的问题是如果一个编译单元中的非局部静态对象的初始化使用了另一个编译单元中的非局部静态对象，那么它所使用的对象可能是未初始化的，因为不同翻译单元中定义的非局部静态对象初始化的相对顺序是未定义的。比如说我们在本地有一个 `FileSystem` 的类，其他客户需要调用这个类

```c++
// 假设有一个FileSystem类，它让互联网上的文件看起来像是本地文件
class FileSystem {
public:
	// ...
	std::size_t numDisks() const; // 众多成员函数之一
    // ...
};
FileSystem tfs;
```

现在假设某些客户在他的本地创建了一个新的类，用于处理文件系统中的目录，很自然它们的类药使用 `tfs` 对象，但是因为 `tfs` 是在其他编译单元的非局部静态变量（全局变量），除非 `tfs` 在  `Directory` 构造函数之前就已经初始化了，否则 `Directory` 会在 `tfs` 初始化之前就使用它，这就会产生问题

```c++
class Directory {
public:
    Directory(params);
    // ...
}

Directory::Directory(params) {
	// ...
    std::size_t disk = tfs.numDisks(); // 使用tfs对象
    // ...
}
```

解决办法是**将非局部静态对象替换为局部静态对象。**将每个非局部静态对象移动到自己的函数中，并将其声明为static。这些函数返回它们包含的对象的引用。客户调用函数而不是引用对象

依据是C++保证，函数内的局部静态对象会在调用该函数时第一次遇到该对象的定义时初始化

```c++
class Directory { /*...*/ }; // 和以前一样

Directory::Directory(params) {
    // ...
    //与之前一样，除了对TFS的引用现在变为tfs (){
	std:::size_t disks = tfs().numDisks();
    // ...
}

Directory& tempDir() { //这将替换tempDir对象，它可以是Directory类中静态函数
	static Directory td; //定义/初始化局部静态对象
	return td; //返回指向它的引用
}
```

注意：在多线程程序中，最好还是在单线程启动阶段手动调用函数来完成初始化

# 构造/析构/赋值运算

## *条款5：了解C++默默编写并调用哪些函数*

### 编译器拒绝生成默认赋值拷贝

在有些情况下编译器会拒绝生成默认的赋值拷贝，比如说下面这个例子

```c++
template<typename T>
class NamedObject {
public:
    NamedObject(std::string &name, const T& value)
        : name_value_(name), object_value_(value)
    {}
    void operator=(const NameObject &rhs) {
        name_value_ = rhs.name_value_;
    }
    // 假设没有自定义operator=
private:
    std::string &name_value_; // 现在是一个引用
    const T object_value_;    // 现在是一个常量
};
```

现在使用这个类

```c++
std::string newDog("Persephone");
std::string oldDog("Satch");
NamedObject<int> p(newDog, 2);
NamedObject<int> s(oldDog, 36);
p = s;   // 编译器会报错
```

在上面的情况中，两个类成员一个是引用，一个是常量。在这种情况下编译器无法生成一个合成赋值拷贝，因为C++不允许引用的重新绑定，也不允许修改常量

若是有引用和常量这种两类成员变量的情况下，一定要自定义operator=来告诉该怎么赋值，比如说我们自定义为只修改 `name_value_`

```c++
class NameObject {
    // ...
    void operator=(const NameObject &rhs) {
        name_value_ = rhs.name_value_;
    }
    // ...
}
```

## *条款6：明确阻止编译器生成代码*

这个条款主要是关于C++11之前禁止拷贝构造生成的（第三版成书时间早于C++11标准，无delete关键字），可以看 *Cpp基础&11.md* 的构造函数-拷贝控制操作-阻止拷贝-C++11之前：声明为private

## *条款7：为多态基类声明virtual析构函数*

这部分在 *Cpp基础&11.md* 的继承-派生类的构造函数-基类虚析构函数中有

总结一下就是多态基类应该声明虚析构函数，非基类或非多态使用的类不应该声明虚析构函数

补充的点是对于纯虚函数（即抽象类），它也要设置为虚析构，且必须提供虚析构的定义，因为要用到

```c++
// Abstract w/o Virtuals
class AWOV {
public:
    virtual ~AWOV() = 0; // 声明为纯虚析构函数
}
AWOV::~AWOV() {} // 必须提供纯虚析构函数的定义
```

## *条款8：别让异常逃离析构函数*

别让异常逃离析构函数 Prevent exceptions from leaving destructors。不要逃离指的是不让异常继续从析构函数中继续throw出去，没有说不能在析构中产生异常，只要try析构，然后catch到处理好就可以

### 场景

从析构函数逃离的异常很难处理

### 解决方案

有两种主要的解决方法

1. 若发生close抛出异常，就终止程序，通常可以调用abort
2. 吞下调用close引起的异常

## *条款9：绝不在构造和析构过程中调用虚函数*

## *条款10：操作符重载返回当前对象*

## *条款11：自我赋值安全性*

## *条款12：要完整拷贝*

# 资源管理

## <span id="条款13">*条款13：以对象管理资源*</span>

## *条款14：RAII对象拷贝*

## *条款15：RAII对象的原始资源访问*

## *条款16：new和delete要一致*

## *条款17：独立构造智能指针*

# 设计与声明

## *条款18：让接口容易被正确使用*

## *条款19：设计class就是设计type*

## *条款20：优先使用常量引用传递*

### 为什么要常量？

值传递 pass-by-value 时是通过对象的拷贝构造来产生一个副本来传入参数的，这种代价可能会很大，特别是对于自定义数据类型

按照约定，内置类型、STL中的迭代器和函数对象都是值传递的

至于传常量有两个好处

1. 明确接口意图，告诉用户传入的参数不会被改变
2. **保护参数不变性**，使用常量引用可以确保在函数内部不会修改传入的参数值
3. **支持常量和非常量**：常量引用可以接受常量和非常量类型的实参。这增加了函数的通用性。这是因为非常量传给const是权限缩小，扩大了传参范围，即const参数和普通参数都可以传给const引用

当然传常量引用的前提是我们没有把参数设计为输出型参数，否则传的是常量引用我们也自然就不能够修改它了

若不做输出型参数的情况下，仍然把常量引用设置为普通引用会产生两个问题

1. 给使用者一种误导，即这个参数在函数内部是可以被改变的，甚至会意外改变这个参数
2. 限制了传参范围，此时const数据类型因为权限扩大问题无法传递给普通参数

### 切割问题

## *条款21：函数返回不要滥用引用*

若函数要返回的对象是在函数内部构造的（非静态）局部对象，那么一旦出了块作用域编译器就自动调用析构把它销毁了，此时若返回引用会产生空指针错误，要用传值返回

如果是在函数作用域里new到堆上的话，还要考虑没有delete会产生内存泄漏的问题

只有当返回的对象是在函数开始之前就已经存在的对象，即对象的作用域不限于函数的块作用域，此时才可以传引用返回

当不希望返回的对象被修改时，返回对常量的引用

## *条款22：将数据设为私有*

## *条款23：非成员非友元函数优先*

## *条款24：参数都需要转换的函数*

## *条款25：不会抛出异常的swap*

# 实现

## *条款26：尽量推迟定义变量*

未使用的变量是有开销的，特别是一些大型的类或数据结构，因此应该尽可能避免过早地定义变量，既可以提高效率，也可以让结构更清晰

### 异常

比如说下面这个例子中过早地定义了变量 `encrypted`，若是中间抛出了异常那么就浪费了构造和拷贝赋值了

```c++
std::string encryptPassword(const std::string &password) {
	std::string encrypted; // 调用默认构造函数对encrypted进行初始化
	encrypted = password; // 调用赋值拷贝给encrypted赋值
    if (password.length() < MinimumPasswordLength) {
        throw logic_error("Password is too short");
    }
    encrypt(encrypted); // 将password加密后放入变量encryptedreturn encrypted;
}
```

将定义变量 `encrypted` 推迟到真正需要的时候

```c++
std::string encryptPassword(const std::string &password) {
    if (password.length() < MinimumPasswordLength) {
        throw logic_error( "Password is too short");
    }
	std::string encrypted(password); // 通过拷贝构造函数定义和初始化
    encrypt(encrypted); // 将password加密后放入变量encrypted
	return encrypted;
}
```

### 循环

循环是一个经典的场景，但是下面两种到底构造、析构和赋值的开销谁大也不好说

* 定义在循环外：1次构造+1次析构+n次赋值

  ```c++
  Widge w;
  for (int i = 0; i < n; i++)
      w = /*some value*/;
  ```

* 定义在循环内：n次构造+n次析构

  ```c++
  for (int i = 0; i < n; i++) 
      Widge w(/*some value*/);
  ```

## *条款27：尽量少做强制转换*

## *条款28：避免返回内部成员的句柄*

## *条款29：为异常安全而努力*

## *条款30：内联函数*

这条条款基本上所有的内容都在 *Cpp基础&11.md* 介绍过了，下面是补充的一些内容

所有的虚函数都不允许内联，因为通过虚函数实现动态绑定的多态

构造函数和析构函数是特殊的，虽然它们看起来都是空的，但实际上不是，要做大量的异常处理和可能需要的资源销毁处理，所以非常不适合内联

## *条款31：降低文件编译依赖*

# 继承与面向对象设计

## *条款32：public继承是is-a关系*

## *条款33：避免遮掩父类的名称*

## *条款34：区分接口和实现继承*

## *条款35：虚函数的替代方案*

## *条款36：非虚函数是静态绑定*

## *条款37：不要重写默认参数值*

## *条款38：*

## *条款39：慎用私有继承*

## *条款40：慎用多重继承*

# 模版与泛型编程

## *条款41：隐式接口和编译器多态*

## *条款42：typename的双重含义*

## *条款43：基类模板名称访问*

## *条款44：抽离无关的模板代码*

## *条款45：成员函数模板*

## *条款46：类模板的友元函数*

## *条款47：萃取类*

## <span id="条款48">*条款48：认识template元编程*</span>

# 定制new和delete

## *条款49：设置new-handler*

## *条款50：自定义new和delete的时机*

## *条款51：遵守new和delete的约定*

## *条款52：定位new*

# 杂项讨论

## *条款53：不要轻忽编译器的警告*

## *条款54：让自己熟悉包括TR1在内的标准程序库*

## *条款55：Boost库*