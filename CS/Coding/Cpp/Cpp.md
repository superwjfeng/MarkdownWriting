# C++对于C缺陷的修正

## *C++版本*

C with classes -> C++1.0 -> ... -> C++98（C++标准第一个版本，引入STL库）-> ... -> C++11（增加了许多特性，使得C++更像是一种新语言）-> ... C++20（自C++11以来最大的发行版，引入了许多新的特性）

## *命名空间 namespace*

在大型的Project中，同一个作用域中可能存在大量同样命名的变量/函数/结构体等，C编译器无法解决这种冲突，C++通过命名空间解决了该冲突

### 定义命名空间

```cpp
namespace wjF {
    int rand = 1; // 定义变量
    int Swap (int* left, int* right) { // 定义函数
            int tmp = *left;
            *left = *right;
            *right = tmp;
    }
    struct Node {// 定义结构体或类
            struct Node* next;
            int val;
    };
}
```

* 命名空间嵌套

```cpp
namespace N1 {
    // 各种定义...
    namespace N2 {
            // 各种定义...
    }
}
```

* 同一个工程中允许多个相同名称的命名空间，编译器最后会合成为同一个命名空间

### 命名空间的展开和使用

* 不展开，加命名空间/指定空间访问

```cpp
// 指定空间访问的操作符为 ::
std::cout << "Hello World!" << std::endl; 
```

* 部分展开

```cpp
using std::cout;
using std::endl;
std::cout << "Hello World!" << std::endl; 
```

* 完全展开

```cpp
using namespace std;
```

* 建议
  * 项目中，尽量不要展开std库。可以指定空间访问+展开常用
  * 日常练习可以展开

## *缺省参数/默认参数 Default Parameter*

### 缺省参数概念

缺省参数是声明或定义函数时为函数的参数指定一个默认值。在调用该函数时，若没有指定实参则采用该形参的默认值，否则使用指定的实参

### 缺省参数分类

* 全缺省参数

```cpp
void Func(int a=10, int b=20, int c=30);
```

* 半缺省参数

```cpp
void Func(int a, int b=20, int c=30);
```

### 注意点

* 半缺省参数必须是位于左边，且不能间隔着给

```cpp
void Func(int a, int b, int c=30); // 正确
void Func(int a=10, int b=20, int c); // 错误，必须是在左边
void Func(int a, int b=20, int c); // 错误，不能间隔着给
```

* 缺省参数不能在函数声明和定义中同时出现。当分离定义时，以声明为准，因为在汇编形成符号表时以声明中的定义为准
* 缺省值必须是常量或者全局变量

## *函数重载 Function Overloading*

### 函数重载概念

* 函数重载允许在同一作用域中声明几个功能类似的同名函数。函数重载很方便，就像在使用同一个函数一样
* 函数重载类型
  * 参数个数不同

    ```cpp
    int func(int a, double b);
    int func(int a, double b, char c);
    ```

  * 参数类型不同

    ```cpp
    int func(int a, char b);
    int func(double a, double b);
    ```

  * 参数类型顺序不同

    ```cpp
    int func(int a, double b);
    int func(double a, int b);
    ```

    * 注意：**只有返回值不同是不构成函数重载的**，因为编译器是根据传递的实参类型推断想要的是哪个函数，所以若只有返回类型不同则无法识别

### 函数匹配

函数匹配 funciton matching/重载匹配 overload resolution

* 编译器找到最佳匹配 best match，并生成调用该函数的代码
* 无匹配错误 no match
* 二义性调用 ambiguous call

### C++支持函数重载的原理 -- 符号修饰Name-decoration/符号改编 Name-mangling

只有声明的函数或变量在本目标文件中是没有分配虚拟地址的，只有在定义之后才会分配内存地址。在编译器的链接过程中，会去找总符号表（同名的cpp文件和其头文件生成一份符号表），里面记录着不同函数的地址。只有声明的函数可以通过定义文件的符号表找到自己的地址，若没有找到就会报链接错误（编译阶段只报语法错误）

在Linux中利用`objdump -S`指令，可以发现在Linux的gcc编译器中，C语言编译器直接用函数名作为其符号表的命名，比如`<Func>`；而C++编译器则会进行函数名修饰，比如分别为`<_Z4Funcid>`和`<_Z4Funcii>`。修饰规则比较复杂，每种编译器在不同系统的修饰规则也不同。关于GCC的基本C++修饰规则可以看自我修养P88

binutils工具包里面了一个 `c++filt` 工具来解析被修饰过的名称

```shell
$ c++filt _Z4Funcid
Func(int, double)
```

符号修饰规则不仅用于函数重载，对于全局（静态）变量和（局部）静态变量也要进行符号修饰防止冲突

因为不同的编译器使用的符号修饰规则是不同的，所以不同的编译器编译产生的ELF文件是无法通过链接器连接到一块的，因为在符号表里找不到需要的符号，这是导致不同编译器之间不能互操作的主要原因之一

### C和C++互相调用库 `extern "C"`

* 静态库.lib和动态库.dll（见操作系统）

* C++调用C库
  * C++中在调用C的头文件时使用 `extern "C"`：告诉C++的编译器，这里面的函数使用C的库实现的，用C的规则去链接查找它们

    ```cpp
    extern "C" {
            #include "....h"
            // ...
    }
    extern "C" int var; //单独声明某个符号为C语言符号
    ```
    
  * 附加库目录
  
    <img src="VS_C_library1.png" width="80%">
  
  * 附加依赖项
  
    <img src="VS_C_library2.png" width="80%">

一种更通用的方式是在系统头文件中添加 `__cplusplus` 条件编译，这样可以让C++能调用C，也能让C调用C++。比如说C语言共享库 `string.h` 中的 `memset` 函数

```cpp
#ifdef __cplusplus
extern "C" {
#endif
void *memset(void *, int, size_t);
#ifdef __cplusplus
}
#endif
```

## *引用 Reference*

注意：这部分特指左值引用

### 引用的概念

引用不是新定义一个变量，而是给已存在变量取了一个别名，**编译器不会为引用变量开辟内存空间，它和它引用的变量共用同一块内存空间**

`&` 操作符：和类型在一起的是引用，和变量在一起的是取地址

### 引用特性

* 引用在定义时必须初始化
* 一个变量可以有多个引用
* 引用一旦已经引用了一个实体后，不能再引用其他实体
* 除了const和类以外，其他的引用必须严格类型匹配

```cpp
int a = 1;
int& b = a; // 必须初始化

int& c = a; // 同一个变量可以有多个引用

int x = 10;
b = x; // 将会同时赋值给b和a，因为b和a是同一个地址
```

### <span id="lvalue">使用场景</span>

* 函数传参
  
  **函数传参本质上和变量的初始化过程是一样的**，都是把一个右值传给左值
  
  * 传值传参 passed by value 需要拷贝。C中常用指针来传递参数，从而使用输出的目的，但是比如在链表中要改变指针自身的时候要传二级指针就很麻烦
  
  * 引用传参 passed by reference/传引用调用 called by reference
  
    * 输出型参数，突破函数只能返回一个值的限制
  
      ```cpp
      typedef struct SeqList {
              //...
      }SL;
      void SLPushBack(SL& s, int x) {
              //...
      }
      int main() {
              SL sl;
              SLPushBack(sl); // 不用传指针了
      }
      ```
  
    * 大对象传参，减少拷贝，提高效率
  
* 做返回值
  * 传值返回（见C语言中的函数栈帧），小对象放寄存器，大对象放上层栈帧
    > 做法是将return的值装载到一个寄存器中，并将寄存器中保存的值给原函数中的接收变量。如果是将临时变量z设置为静态变量z，即 static int z。那么z会被保存到静态区中，并不会被销毁。但编译器仍然会选择将z的值加入到寄存器中生成临时拷贝后返回给上层
  * 传引用返回的问题
    * ret的结果是未定义的，栈帧调用结束时，系统会清理栈帧并置成随机值，那么这里ret的结果就是随机值。因此该程序使用引用返回本质是不对的，越界后结果没有保证。因此传引用返回的前提是**出了函数作用域，返回对象就销毁了，那么一定不能用引用返回，一定要用传值返回**。
    * 修改方式
      * 将Count中的int放到静态区中，这样Count调用结束，栈帧销毁后，静态区中的int也不会被销毁
      * malloc出来的内存是在堆上也不会被销毁，因此传引用返回可以应用到顺序表等数据结构中提高效率

        ```cpp
        // 错误
        int& Count() {
                int n = 0;
                n++;
                // ...
                return n;
        }
        // 放到静态区中传引用返回
        int& Count() {
                static int n = 0;
                n++;
                // ...
                return n;
        }
        int main() {
                int ret = Count();
                return 0;
        }
        ```
    
  * 传引用返回的优势
    * 输出型返回对象，调用者可以修改返回对象，比如 `operator[]`
    * 减少拷贝

### 例子：传引用与传值在类中的应用

```cpp
// 实例
#include <iostream>
using std::cout;
using std::endl;
class A {
public:
    A(int a = 0) {
        _a = a;
        cout << "A(int a = 0)->" << _a << endl;
    }
    // A aa2(aa1);
    A(const A& aa) {
        _a = aa._a;
        cout << "A(const A& aa)->" << _a << endl;
    }
    ~A() {
        cout << "~A()->" << _a << endl;
    }
private:
    int _a;
};

void func1(A aa) {} //传值传参
void func2(A& aa) {} //传引用传参

A func3() {
    static A aa(3);
    return aa;
}
A& func4() {
    static A aa(4);
    return aa;
}
```

* 中间临时量 temporary object 问题，具体看《程序员的自我修养--链接、装载与库》10.2.3函数返回值传递机制
  
  * 函数的返回对于1-8字节的小对象，直接通过eax寄存器存放的临时量返回。注意这个临时对象编译器默认将其设置为const
  * 对于大于8字节的大对象，会在上层栈帧开辟temp空间进行返回
  
* 输入

  <img src="传值输入与传引用输入的比较.png" width="50%">

  * 传值输入：需要调用拷贝函数，压栈时创建临时的形参，将实参拷贝给形参后调用形参，调用拷贝构造函数有栈帧开销和拷贝临时量的空间浪费，从上图试验中可以看出，传值需要多调用拷贝和析构，开销很大
  * 传引用输入：不需要调用拷贝构造函数，直接传递引用
  * 结论：传引用和传值对于内置类型效果不明显，都需要4个或8个字节。但对自定义类型效果明显，传引用不需要调用拷贝构造进行拷贝 

* 输出

  <img src="传值输出与传引用输出的比较.png">

  * 传值输出：需要调用拷贝构造函数拷贝生成一个**临时量**，然后将临时量的值返回给外层函数的接收变量，栈帧销毁后一块销毁，即传值返回不会直接返回原来栈帧中的对象，而是返回对象的拷贝。调用拷贝构造函数有栈帧开销和拷贝临时量的空间浪费
  * 传引用输出：不需要调用拷贝构造函数，直接传递引用
  * 结论
    * 调用拷贝构造函数有栈帧开销
    * 自定义类型有时需要深拷贝
    * 自定义类型往往比较大

### 编译器对传值传参和传引用传参的优化

这部分在C++2022.08.05的课程有讲

编译器在release下的优化是取消了中间临时变量，直接将要返回的值拷贝给接收变量，Linux可以通过 `-fno-elide-constructors` 关闭编译器的所有优化

但是不能因此而取消中间临时变量，因为根据建立栈帧的顺序，ret必须要紧跟着func栈帧才能被找到，并且进行优化

```cpp
int ret1 = func(x); //只有直接用变量来接收编译器才会进行优化
int ret2;
// ...
ret2 = func(y); //这时候就不会优化的
```

<img src="返回的编译器优化.png">

### 指针与引用的关系

* 指针和引用用途基本是相似的，但指针更强大也更危险
* 使用场景
  * 链表定义中不能使用引用，因为引用必须要初始化赋值且引用不能更改指向
* 语法特性及底层原理
  * 语法角度来看引用没有开空间，指针开了4或8字节空间
  * 底层原理来看，引用底层是用指针实现的

## *内联函数 Inline*

* 定义：以 `inline` 修饰的函数叫做内联函数，编译时C++编译器会在调用内联函数的地方展开，没有函数调用建立栈帧的开销，而是直接替换成了一些机器代码，因此内联函数提升程序运行的效率
* 使用场景：堆排序和快速排序中需要被频繁调用的Swap函数

### C语言中用宏函数来避免建立和销毁栈帧

* 宏的优点：复用性变强、宏函数提高效率，减少栈帧开销、不用进行传入参数的类型检查（用enum）
* 宏的缺点：可读性差，复杂不易编写、没有类型安全检查、不方便调试（预处理阶段就被替换掉了）

```cpp
#define ADD(a, b) ((a) + (B))
// 记住两个特殊场景
ADD(1, 2) * 3;
ADD(a | b, a&b);
```

### inline特性

* inline是一种以空间换时间的做法，若编译器将函数当成内联函数处理，在编译阶段，会用函数体替代函数调用。缺陷是可能会使文件目标变大；优势是少了调用开销，提高程序运行效率
* inline对于编译器而言只是一个建议，不同编译器关于inline实现机制可能不同
* inline不建议声明和定义分离，分离会导致链接错误。inline不会被放进编译生成的符号表里，因为inline定义的函数符号是不会被调用的，只会被展开使用。推荐声明+定义全部写在头文件中（这也符合模块化设计声明和定义分离的设计初衷，便于使用）

## *C++的强制类型转换*

### 为什么C++需要四种类型转换

C++继承了C语言的隐式类型转换和显式类型转换体系，可以看C.md类型转换部分

这里有一个经典的错误可以看Cpp.md的string类模拟实现部分的insert部分：[经典隐式类型转换错误](#经典隐式类型转换错误)

C++兼容了C隐式类型转换和强制类型转换，但希望用户不要再使用C语言中的强制类型转换了，希望**使用规范的C++显式强制类型转换**

强制类型转换的形式为 **`cast_name<type>(expression);`**

### `static_cast`

```cpp
double d = 12.34;
int a = static_cast<int>(d);

void *p = &d;
double *dp = static_cast<double*>(p); //规范使用void*转换
```

任何具有明确定义的类型转换，只要不包含底层const，都可以使用 `static_cast`

`static_cast` 是一种**安全**的类型转换运算符，它可以将一种类型转换为另一种类型。`static_cast` 可以执行隐式类型转换，例如将整数类型转换为浮点类型，也可以执行显式类型转换，例如将指针类型转换为整数类型。`static_cast` 进行类型转换时会执行一些类型检查和转换，以确保类型转换是合法的。

### `reinterpret_cast`

C++类型转换之reinterpret_cast - 叫啥名呢的文章 - 知乎 https://zhuanlan.zhihu.com/p/33040213

`reinterpret_cast` 可以将一个指针或引用类型强制转换为另一个指针或引用类型，而**不进行任何类型检查或转换**。`reinterpret_cast` 主要用于以下情况：

1. 将指针或引用类型转换为另一个指针或引用类型，以便可以在它们之间进行转换
2. 在某些特殊情况下，当需要使用指针或引用类型表示不同的对象类型时，可以使用 `reinterpret_cast` 进行类型转换。这通常发生在涉及底层硬件或操作系统接口的代码中
3. 在某些情况下，`reinterpret_cast` 也可以用于类型擦除，即将模板类型擦除为一个没有模板参数的类型，以便可以在运行时处理它们

`reinterpret_cast`的危险性 ⚠️：需要注意的是，使用 `reinterpret_cast` 进行类型转换时必须非常小心。由于它不执行任何类型检查或转换，如果类型转换不正确，可能会导致未定义的行为或错误的结果。因此，应该尽可能避免使用 `reinterpret_cast`，而优先考虑使用其他更安全的类型转换运算符，例如 `static_cast` 或 `dynamic_cast`

### `const_cast`

通过 `const_cast` 去除const属性 cast away the const，但是在去除const属性后再进行写就行未定义的行为了

```cpp
const char *pc;
char *p = const_cast<char*>(pc); //pc现在是非常量了
```

`const_cast` 常用于有函数重载的上下文中

### `dynamic_cast`

`dynamic_cast`和前面的三类用于增强C++规范的类型转换不同，它专用于区分父类指针是指向子类还是指向父类，即用于将一个父类对象的指针/引用转换为子类对象的指针或引用，即动态转换

父类对象是无论如何都不允许转子类的，但允许父类的指针或引用转换为子类对象的指针或引用

* 向上转型：子类对象指针或引用指向父类对象或引用，也就是切片，不需要进行转换

  ```cpp
  class A {};
  class B : public A {};
  B bb;
  A aa1 = bb;
  A& ra1 = bb; //向上转换切片，所以没有产生中间变量，也就不需要const了
  ```

* 向下转型：父类对象指针或引用指向子类对象或引用

  * 父类指针强制转子类有越界的风险，因为指针相当于规定了能看到的内存空间为多大，而父类的大小小于等于子类
  * 只能用于父类含有虚函数的多态类
  * 这时候只有使用 `dynamic_cast` 才是安全的。安全是什么意思？
    * 如果指针是指向子类，那么可以转换是安全的
    * 如果指针是指向父类，那么不能转换，转换表达式返回nullptr

### RTTI思想

C++是一种静态类型 statically typed 语言，其含义是在编译阶段进行类型检查 type checking。对象的类型决定了对象所能参与的运算，因此类型检查的含义是编译器负责检查类型是否支持要执行的运算。当然前提是编译器必须要知道每一个实体对象的类型，所以这要求在使用变量前必须要声明其类型

Run-time Type Identification 运行时类型识别

C++通过以下方式来支持RTTI

* `typeid` 运算符：获取对象类型字符串
* `dynamic_cast` 运算符：父类的指针是指向父类对象还是子类对象
* `decltype` 运算符：推导一个对象类型，这个类型可以用来定义另一个对象

## *C++的IO流*

### C语言的IO函数

* `scanf()/printf()` 处理终端/控制台IO
* `fscanf()/fprintf()` 处理文件IO
* `sscanf()/sprintf()` 处理字符串IO

C语言是面向对象的，只能指针内置类型

### C++IO流设计

什么是流 Stream？流是对有序连续且具有方向性的数据的抽象描述。C++流是指信息从外设向计算机内存或从内存流出到外设的过程

为了实现这种流动，C++实现了如下的标准IO库继承体系，其中 `ios` 为基类，其他类都是直接或间接派生于它

<img src="IO_Libraray.png" width="80%">

* `<istream>` 和 `<iostream>` 类处理控制台终端
* `<fstream>` 类处理文件
* `<sstream>` 类处理字符串缓冲区
* C++暂时还不支持网络流的IO库

### C++标准IO流

C++标准库提供给了4个**全局流对象** `cin`，`cout`，`cerr`，`clog`。从下图可以看到 `std::cin` 是一个 `istream` 类的全局流对象

<img src="cin对象.png">

C++对这部分的设计并不是特别好，因此这几个对象在输出上基本没有区别，只是应用该场景略有不同

使用 `cin` 进行标准输入即数据通过键盘输入到程序中；使用 `cout` 进行标准输出，即数据从内存流向控制台(显示器)；使用 `cerr` 用来进行标准错误的输出；使用 `clog` 进行日志的输出

* `cin` 为缓冲流，相当于是一个 `char buff[N]`对象，键盘输入的数据保存在缓冲区中，当要提取时，是从缓冲区中

* 空格和回车都可以作为数据之间的分隔符，所以多个数据可以在一行输入，也可以分行输入。但若是字符型和字符串，则空格（ASCII码为32）无法用 `cin` 输入。字符串中也不能有空哦个，回车符也无法读入

  ```cpp
  // 2022 11 28
  // 输入多个值，默认都是用空格或者换行分割
  int year, month, day;
  cin >> year >> month >> day;
  scanf("%d%d%d", &year, &month, &day);
  scanf("%d %d %d", &year, &month, &day); //不需要去加空格
  
  //20221128 无空格
  scanf("%4d%2d%2d", &year, &month, &day); //scanf可以通过加宽度直接处理
  //cin在这种情况下反而会比较麻烦
  string str;
  cin >> str;
  year = stoi(str.substr(0, 4));
  month = stoi(str.substr(4, 2));
  day = stoi(str.substr(6, 2));
  ```

* `cin, cout` 可以直接输入和输出内置类型数据，因为标准库已经将所有内置类型的输入和输出全部重载了

  <img src="ostream.png" width="80%">

* 对于自定义类型，可以重载运算符 `<<` 和 `>>`

  <img src="流输入运算符重载.png">

  如上图是对string类的 `>>` 运算符重载，意思是支持了 `istream& is` 和 `string& str` 之间的 `>>` 运算

  这部分可以参考Date类中的 `<<` 重载

* OJ中的输入与输出

  ```cpp
  // 单个元素循环输入
  while(cin>>a) {
      // ...
  }
  // 多个元素循环输入
  while(c>>a>>b>>c) {
  	// ...
  }
  // 整行接收
  while(cin>>str) {
  	// ...
  }
  ```

  OJ中有时候可能会出现用 cin 和 cout 效率过不了的问题，可以考虑采用 `printf` 和 `scanf`。这是因为C++为了要兼容C语言，内部需要采取和C语言输入输出函数同步顺序的一些处理，因此可能会导致效率的下降

  因此还是建议在OJ中用 `printf` 和 `scanf`

### `istream` 类型对象转换为逻辑条件判断值

考虑有多组测试用例的情况，比如上面的代码中输入多组日期，那么要用

```cpp
char buff[128];
while (scanf("%s", buff) != EOF) {}
// or
while (cin >> str) {} //如何实现逻辑判断？
```

对于 `scanf()` 而言，这是一个函数调用，返回的是接收到的值的数目，因此很好理解，若写入失败（满了）或按 `ctrl+z`，就返回EOF，ASCII码中 `EOF==0`，因此while循环会被终止，可是 cin 是一个全局流对象，为什么它也等价于一个逻辑值，从而可以作为while循环的判断条件呢？

内置类型是可以隐式类型转换成自定义类型的，编译器会自动调构造和拷贝构造（有些编译器自动优化为只有自定义），反过来自定义类型也可以转换为内置类型，但是此时需要显式重载 `operator TYPE()`，比如重载 `operator bool()` 将自定义类型转换成bool内置类型

`cin >> str` 的返回值是 `istream` 对象，因此在istream类内部实现了 `operator bool()` ，因此当 `while (istream& obj)` 可以进行逻辑判断

`operator bool()` 全部是复用的ios基类的实现

### C++文件IO流

```cpp
ifstream ifs("test.cpp");
while (ifs) { //重载了operator bool()
    char ch = ifs.get();
    cout << ch;
    ////或者下面这么写会自动忽略空格
    //char ch;
    //while (ifs >> ch) {
    //    cout << ch;
    //}
}
```

### stringstream

# 类与对象（上）-- 类的大框架

## *类的定义*

```cpp
// .h 文件中的声明
class className {// 类名
    // class默认成员访问权限为private
    int attribute = 1; // 类属性和方法用 "." 操作符调用，即 className.attribute
    void Method_declaration(); // 定义为类的一部分的函数称为成员函数 member fucntion 或 类方法 method
}
// .cpp 文件中的类方法定义
void className::Method_definition() {/*...*/}
```

### 类的定义原则

* 类函数的声明必须要在类内部，而类函数的声明则既可以在类内部也可以在类外部
* 小函数若想成为inline，直接在类里面定义即可。但具体是否会作为inline处理取决于编译器
* 若是大函数，应该声明和定义分离。一般情况下都使用这种定义方式

### 类的作用域

变量搜索遵循局部优先，因此在c++文件中定义类函数时要用 `::` 指定类的作用域

类里面用 `typedef` 或者 `using` 起别名，但要注意用来定义类型的成员必须先定义后使用

### 类的命名规范

* 单词和单词之间用驼峰法
* 函数名、类名等所有单词首字母大写 `DataMgr`
* 变量首字母小写，后面单词首字母大写 `dataMgr`
* 类的属性变量在开头或结尾使用 `_`（STL库） 或者驼峰（公司规范居多）`_dataMgr`

```cpp
class Date {
public:
    void Init(int year)
        _year = year;
        // 若属性不写成_year，则可能会出现 year = year 的混淆
private:
    int _year;
}
```

## *类的访问限定符及封装*

### 访问限定符 Access Modifier

```cpp
class A {
public:
    void PrintA()
        cout << _a << endl;
private:
    char _a;
};
```

* 分类
  * public 公有
  * protected 保护
  * private 私有
* 访问限定符的说明
  * public修饰的成员在类外可以直接访问
  * protected和private修饰的成员在类外不能直接被访问
  * 访问权限作用域从该访问限定符出现的未知开始直到下一个访问限定符出现时为止
  * 若后面没有访问限定符，作用域就到 `}` 为止
  * 三者都是只对类作用域外有效，即类内无论是什么限定符都可以互相取
  * struct定义的类默认访问权限是public，而class定义的类默认访问权限则是private
* C++中struct和class的区别：C++需要兼容C语言，所以C++中struct可以当成结构体使用。但C++中的struct既可以定义变量，也可以定义函数。因此C++中将struct升级成了class。区别是struct定义的类默认访问权限是public，而class定义的类默认访问权限则是private。

### 封装 Encapsulation

隐藏对象的属性和实现细节，即隐藏成员变量。仅对外公开接口来和对象进行交互（即开放成员函数接口）

```cpp
cout << st.a[st.top] << endl; // 这样调栈顶数据是错误的，因为类属性受保护。
// 而且很有可能出错，因为使用者并不知道底层的实现，top到底指向哪一个数据？
cout << st.Top() << endl;  // 这样是正确的，要使用给的函数接口
```

C语言没办法封装，可以规范的使用函数访问数据，也可以不规范的直接访问数据；C++使用封装，必须规范使用函数访问数据，不能直接访问数据

注意下面一个例子，在类内声明，在类外定义也是可以无视限定符的

```cpp
class A {
public:
    void Print(); //类成员声明
private:
    int _a;
};

void A::Print() { 
    cout << _a << endl; 
} //类成员定义
```

## *类的实例化 Instantiation*

* 类的声明不会占用物理空间
* 一个类可以实例化出多个对象，实例化出的对象占用实际的物理空间，存储类成员变量

## *类对象模型*

### 类对象的可能存储方式

* [ ] 对象中包含类的各个成员：问题是每个对象中成员变量都需要调用同一份函数，导致大量的冗余代码
* [ ] 实例化的每个对象成员变量都是独立空间，是不同变量。但是每个对象调用的成员函数都是同一个。没有被采用，在虚表和多态部分采用
* [x] 公共代码区
  * 只保存**非静态的成员变量**，成员函数存放在公共的代码段
  * 编译链接时就根据函数名去公共代码区找到函数的地址，然后call函数地址
  * 下面代码可以顺利运行，说明没有对空指针解引用，所以实际使用的是公共代码区

    ```cpp
    class A {
    public:
        void func() {
            cout << "void A::func()" << endl; 
        }
    };
    
    int main() {
        A* ptr = nullptr;
        ptr->func();
        return 0;
    }
    ```

### 类的内存对齐规则

* 属性和C语言中自定义类型的对齐规则一样
* 方法属于公共区代码，所以不计入类的大小。但是即使属性和方法都为空，也会为其分配1字节占位。不存储实际数据，标识对象存在

## *this指针*

### this指针的引出

```cpp
class Date {
public:
    void Init(int year, int month, int day) {
        _year = year;
        _month = month;
        _day = day;
    }
    void Print() {
        cout <<_year<< "-" <<_month << "-"<< _day <<endl;
    }
private:
    int _year;
    int _month;
    int _day;
    int a;
}
```

```cpp
Date d1;
d1.Init(2022, 7, 17);
Date d2;
d2.Init(2022, 7, 18);
d1.Print();
d2.Print();
```

产生两个对象d1和d2，但是方法中并没有区分作用的对象是哪一个，函数是怎么知道该设置哪一个对象呢？

类中的方法会在第一个形参处放置一个额外的隐式this指针用来访问调用它的那个对象，当然下图中右边的类方法是不对的，因为不允许显式地给出this指针，这里只是为了说明用

编译器负责把调用者的地址传给this

<img src="this_pointer.png">

### this指针的特性

* this指针是所有**非静态**成员函数的第一个隐形形参
* this指针是一指针常量：语法规定不能修改this指针（`T* const this`），因为this指针是被const保护的只读，但可以修改this指针指向的内容
* 也不能在实参和形参位置显式地传递this指针。但可以在类函数的内部使用（不使用也会自动加）
* this指针本质上是成员函数的形参，当对象调用该成员函数时，将对象地址作为实参传递给this形参，所以对象中不存储this指针。VS下将this指针优化到了寄存器中，因为在频繁使用时可以提高效率
* 问题：this指针可以为空吗？可以！参考下面两段对比代码

    ```cpp
    class A {
    public:
        void Print()
            cout << "Print()" << endl;
    private:
            int _a;
    };
    
    int main {
        A* p = nullptr;
        p->Print();
        return 0;
    }
    ```
    
    不会崩溃，虽然p是空指针，但并没对它解引用，因此程序正常运行

    ```cpp
    class B {
    public:
        void Print()
            // 相当于是 cout << this->_b << endl;
            cout << _b << endl;
    private:
            int _b;
    };
    
    int main {
        B* p = nullptr;
        p->Print();
        return 0;
    }
    ```
    
    程序会崩溃，因为对空指针解引用了！

## <span id="const成员函数">*const成员函数*</span>

### 介绍

this指针是一个 `T* const this` 的指针常量，我们不能修改this指针本身，也就说不能莫名其妙把调用者改成其他人。但是仍然可以通过this指针来修改调用者的属性，比如通过 `this->_year` 就可以获取 `_year` 属性了

那么如果我们想保持类属性不变怎么办呢，一个方法当然是可以直接把类属性设置为const，但是这样任何其他的方法也就不能修改属性了，不够灵活。可以通过将某个特定方法设置为const，即const成员函数 const member function 来达成，此时的this指针就相当于变成了 `const T* const this` 

### 权限放大问题

```cpp
class Date {
Public:
    // ...
    void Date::Print() {
        cout << _year << "/" << _month << "/" << _day << endl;
    }
    bool Date::operator<(const Date& d) {
        return !(*this >= d);
    }
}
int main() {
    Date d1(2022, 7, 25);
    const Date d2(2022, 7, 26);

    d1.Print();
    //  d2.Print(); // 编译报错，权限放大
    d1 < d2;
    //  d2 < d1; // 编译报错，权限放大
}
```

* `Date* const this` this指针是一个指针常量，指针本身不能被修改，但可以修改指针指向的内容。当传入一个const Date d2 常变量时，权限就被放大了

* `d2 < d1`，因为 `<` 运算符重载时，第一个参数是this指针常量，和上面的错误是一样的，都是权限放大

* 若要支持以上的调用，我们必须将this指针设置为常量指针。`void Date::Print() const` 修饰的this指向的内容，也就是保证了成员函数内部不会修改成员变量const对象和非const对象都可以调用这个成员函数（权限平移）

  ```cpp
  void Date::Print() const {
      cout << _year << "/" << _month << "/" << _day << endl;
  }
  bool Date::operator<(const Date& d) const {
      return !(*this >= d);
  }
  ```

# 类与对象（中）-- 默认成员函数

## *类的6个默认成员函数*

对于一个空类，编译器会自动生成6个不会显示实现的默认成员函数

* 初始化和清理
  * Constructor 构造函数主要完成初始化工作
    * 大部分类都不会让编译器默认生成构造函数，都要自己写。显式地写一个全缺省函数，非常好用
    * 特殊情况下才会默认生成。比如用两个栈生成一个队列时
    * 每个类最好都要提供默认构造函数
  * Destructor 析构函数
    * 申请了内存资源的类需要显式写析构函数，如Stack, Queue，否则会造成内存泄漏
    * 不需要显式地写析构函数
      * 一些没有资源需要清理的类比如Date
      * 或者是MyQueue这样的类中会调用自定义结构的默认析构函数完成清理
* 拷贝复制
  * Copy constructor 拷贝构造
    * 和析构一样，若申请了内存资源则需要显式定义
    * 不需要显式地写拷贝构造
      * 如Date这种默认生成会完成浅拷贝
      * 或者是MyQueue这样的类中会调用自定义结构的默认拷贝复制函数完成任务
  * Assignment operator overloading 赋值运算符重载：和拷贝构造一样也会面临自定义结构的浅拷贝问题，因此自定义结构也需要显式定义
* 取地址即const取地址重载：主要是普通对象和const对象取地址，这两个很少会自己实现，默认生成的够用了

## *构造函数 Constructor*

### 概念

* 构造函数自动初始化成员属性。构造函数的名字与类名相同，没有返回值。创建类类型对象时由编译器自动调用，以保证每个数据成员都有一个合适的初始值，并且在对象整个生命周期内只调用一次
* 构造函数的任务不是构造对象和开辟栈帧空间，而是对实例化对象进行初始化

### 特性

```cpp
// 函数重载，给默认值
Date() {
    _year = 1;
    _month = 0;
    _day = 0;
}
Date(int year, int month, int day) {
    _year = year;
    _month = month;
    _day = day;
}
// 调用
Date d1(2022, 9, 15);
Date d2; // 赋默认值
---------------------------------------------------------
// 或者是直接给缺省参数，这样实现比较好
Date(int year=1, int month=0, int day=0) {
    _year = year;
    _month = month;
    _day = day;
}
```

* 函数名与类名相同
* 无返回值
* 对象实例化时编译器自动调用对应的构造函数
* 构造函数可以重载
* 因为构造函数无论如何要取得类属性的控制权，所以它**不能被声明为const**

### 定义构造函数

**若没有定义，就自动生成空的默认构造；若定义了非默认构造就不会生成默认构造，必须手动定义一个默认构造，或者用 `default` 强制生成**

* 若类中没有显式定义构造函数，则C++编译器会自动生成一个**无参的默认构造函数**（空默认构造）default constructor，一旦用户显式定义编译器将不再生成
  * C++把类型分成了两类
    * 内置类型/基本类型：int, double, char, pointer, ...
    * 自定义类型：struct, class, ...
  * **C++设计的缺陷**：默认生成的构造函数对**内置类型不做处理**；而自定义类型成员会去调用它的默认构造函数

    ```cpp
    class Time {
    public:
    Time() {// 构造函数
        cout << "Time()" << endl;
        _hour = 0;
        _minute = 0;
        _second = 0;
    }
    private:
        int _hour;
        int _minute;
        int _second;
    };
    class Date {
    public:
        // 没有自己写的构造函数，编译器会自动生成
        void Print()
            cout << _year << " " << _month << " " << _day << endl;
    private:
        int _year; // 内置类型
        int _month; // 内置类型
        int _day; // 内置类型
        Time _t; // 构造类型
    };
    // 用默认构造函数调用
    Date d;
    ```
    
    <img src="constructor_bug.png">
    
    通过调试可以发现，内置类型的 `_year, _month, _day` 都是随机值，没有被初始化，但自定义类型 `_t` 被初始化了
    
    * 这个缺陷在C++11中打了**补丁**：可以为内置类型成员变量在类中声明时给默认值（缺省参数）来修正
    
    ```cpp
    // 一定要注意，这个不是给初始值，而是为了修正构造函数不处理内置类型的缺陷给的缺省值！！！
    private:
        int _year=1;
        int _month=0;
        int _day=0;
        Time _t; // Time* _t 指针也是内置类型，不会被初始化
    ```

* 默认构造函数并不仅仅指默认生成的构造函数，而是有**三类**都可以称为默认构造函数。其特点是**不传参数就可以调用的，且只能存在一种**
  * 我们不写，由编译器自动生成的
  * 我们自己写的全缺省构造函数
  * 我们自己写的无参构造函数（空参数列表）
  * 若写了一个其他的非默认构造函数，那么一定要自己实现上面的三个默认构造函数之一。因为编译器不会自动生成默认构造函数，此时编译器会报如下的错误
    > **没有合适的默认构造函数可用**

## *析构函数 Destructor*

### 概念

析构函数与构造函数功能相反，析构函数不是完成对对象的销毁。局部对象销毁工作是由编译器来完成的。而对象在销毁时会自动调用析构函数，完成对象中资源的清理工作

### 特性

* 析构函数名是在类名前加上字符 `~`
* 无参数无返回值类型，无参数所以析构函数不能重载
* 一个类只能有一个析构函数。若无显式定义，系统会自动生成默认的析构函数
* 对象生命周期结束时，C++编译系统自动调用析构函数
* 若析构函数中没有很多工作时，有些编译器会把它优化掉
* 默认生成的析构函数和构造函数类似
  * 内置类型不处理，但这里不是设计bug，而是系统无法处理。因为系统无法判断当前指针是否是动态开辟内存出来的还是有其他作用
  * 自定义类型成员回去调用它的析构

    ```cpp
    class Stack {
    public:
        Stack(int capacity = 4) {
            _array = (DataType*)malloc(sizeof(DataType) * capacity);
            if (NULL == _array) {
                    perror("malloc fail!\n");
                    return;
            }
            _size = 0;
            _capacity = capacity;
        }
        void Push(DataType data) {
            _array[_size] = data;
            _size++;
        }
        ~Stack() {// Destructor
            cout << "~Stack()->" << _array << endl;
            free(_array);
            _size = _capacity = 0;
            _array = nullptr;
        }
    private:
        DataType* _array;
        int _size;
        int _capacity;
    };
    class MyQueue {
    public:
        void Push(int x){}
    private:
        size_t _size = 0;
        Stack _st1;
        Stack _st2;
    };
    // 调用
    int main {
        Myqueue q;
        return 0;
    }
    ```
    
    <div align="center"><img src="destructor_result.png"></div>
    
    在析构函数 `~Stack()` 中打印了一下，可以发现在自定义类型Stack中调用了两次析构函数
* 若类中没有动态开辟内存时，析构函数可以不写，直接使用编译器生成的默认析构函数；由动态开辟内存时则一定要写，否则会造成内存泄漏

## *拷贝构造函数 Copy constructor*

### 概念

```cpp
int a = 1;
int b = a; // 普通变量的复制

Date d1(2022, 7, 23);
Date d2(d1); // 类对象的拷贝
Date d3 = d1; // 也可以这么拷贝

// 拷贝构造函数
Date(const Date& d) {
    _year = d._year;
    _month = d._month;
    _day = d._day;
}
```

和普通变量一样，有时候想要创建一个一摸一样的新对象。为达成这种性质，需要给类添加拷贝构造函数

### 特性

* 拷贝构造函数是构造函数的一个重载形式

* 拷贝构造函数的参数只有一个且必须是类类型对象的引用，使用传值方式编译器直接报错，因为会引发无穷递归调用

    > illegal copy constructor: first parameter must not be a 'Date'

  * 以传值方式传参需要开辟临时空间，从而拷贝传入的参数
  
  * 但是为了拷贝传入的参数，本身又需要调用拷贝构造函数，从而形成了无穷递归
    * 先做下图试验，通过调试可以发现：对于两个普通的函数 `get1, get2`，当其实参采用类传值传参时会调用类的拷贝构造函数，而传引用则不会。这说明为了创建类实参的临时拷贝，本身就需要调用类的拷贝构造函数
    
      <img src="传值传参试验.png" width="60%">
    
    * 因此有 <img src="copy_constructor_iteration.png" width="80%">
    
    * 为什么不采取传指针？
    
  * 添加 `const` 将引用设置为只读，防止误操作对实参的改变
  
* 若未显式定义，编译器会生成默认的拷贝构造函数。默认的拷贝构造函数对象将内存存储按字节序完成拷贝，这种拷贝叫做浅拷贝/位拷贝 shallow copy/bit-wise copy
  * 对于内置类型不显式定义也可以，但若有内存开辟则浅拷贝存在以下问题
    * 一个对象修改会应先给另一个对象
    
    * 会析构两次，即free同一块内存空间两次，程序崩溃
    
      <img src="浅拷贝问题.png" width="59%">
    
  * 解决方法：自己实现深拷贝 Deep copy/值拷贝 Member-wise copy，深拷贝的实现见string章节

### 拷贝构造函数的典型调用场景

* 使用已存在对象创建新对象
* 函数参数类型为类类型对象
* 函数返回值类型为类类型对象

## *赋值运算符重载 Assignment Operator Overloading*

以Date类的实现为例，详细实现见 Lecture6_class4_date_20220724

### 运算符重载 Operator Overloading

* 运算符重载是为了让自定义类型对象也可以使用运算符像内置对象那样进行运算
* 除了 `.* :: sizeof ?: .` 5个运算符外其他的运算符都可以被重载

### 赋值运算符重载

```cpp
Date& operator=(const Date& d) {
    // this出了类域还存在
    if (this != &d) {// 避免自己给自己赋值
        _year = d._year;
        _month = d._month;
        _day = d ._day;
    }
    return *this; // 返回this指向的值为了支持连续赋值 d2 = d1 = d3
}
```

* 赋值运算符重载的特性
  * 参数类型：const Date& 传引用提高效率，且不修改Date类，因此用const保护
  * 返回值类型：返回引用可以避免传值拷贝，提高返回效率，有返回值的目的是为了支持连续赋值。如果传指针返回，外面接受还需要解引用，非常别扭
  * 要检查是否自己给自己赋值（不写也不会报错，只是为了提高效率）
  * 返回*this：要符合连续赋值的含义
* 赋值运算符只能重载成类的成员函数不能重载成全局函数。赋值运算符如果不显式实现，编译器会生成一个默认的。此时用户再在类外自己实现一个全局的赋值运算符重载，就和编译器在类中生成的默认赋值运算符重载冲突了，故赋值运算符重载只能是类的成员函数
* 用户没有显式实现时，编译器会生成一个默认赋值运算符重载，进行浅拷贝，会产生和默认拷贝构造函数的浅拷贝一样的问题。若有自定义类型成员就必须要显示实现

### 其他运算符重载的复用实现

* 几乎所有的类的运算符重载都只需要自己实现 `= == >` 或者 `= == <`，其他的比较运算符重载（不是加减乘除！）都可以复用这几个运算符
* Date类的 `+= =`复用 （`-= =` 同理）
  * 用+=复用+
  * 用+复用+=：缺点式拷贝次数过多，一直在调用+中的=

    ```cpp
    Date& Date::operator+=(int day) {
        // 要改变自身，所以可以直接对自身操作，采用传引用返回
        if (day < 0)
            return *this -= -day;
        _day += day;
        while (_day > GetMonthDay(_year, _month)) {
            _day -= GetMonthDay(_year, _month);
            ++_month;
            if (_month == 13) {
                _year++;
                _month = 1;
            }
        }
        return *this;
    }
    Date Date::operator+(int day) const {
        // Date ret(*this); // 拷贝构造
        Date tmp = *this; // 注意这里也是拷贝构造，两个已经存在的值才是赋值。两种写法等价
        tmp += day; // 用 += 复用
        return tmp; // 传值拷贝构造
    }
    ```
    
    根据加法的性质，两数相加会生成临时变量，并将临时变量赋给接收值，因此需要新建一个tmp临时变量并采用传值返回。同时拷贝构造所产生的临时变量在栈帧销毁后也被销毁了，所以只能传值返回，不能传引用，否则会引起野指针问题
  
* 一个类需要重载哪些运算符要看哪些运算符对这个类型有意义，比如Date类除法和乘法就没有意义
* 前置++和后置++重载：特殊处理，使用函数重载，默认无参为前置，有参为后置

    ```cpp
    Date& operator++() {// 前置
        *this += 1;
        return *this;
    }
    Date operator++(int) {// 后置，传的参数没有效果，仅仅是用来区分
        Date tmp(*this);
        *this += 1;
        return tmp;
    }
    ```
    
    复用+=，因为前置++会直接改变\*this，因此可以传引用返回，而后置++是先用再++，因此要保留一个\*this的备份tmp，所以要传值返回tmp

### 赋值运算符重载和拷贝构造的区分

```cpp
class A;
A a1;
A a2(a1); // 拷贝构造
A a3 = a1; // 拷贝构造
A a4;
a4 = a1; // 赋值
```

拷贝构造是针对还没有定义的对象的，而赋值是已经存在的对象

### 运算符重载和函数重载的应用（以cout为例）

回顾C++IO流的内容，`<<` 流输出运算和 `>>` 流输入，因为C++库里写好了运算符重载。`cout`对象能做到自动识别类型，这是因为所有内置类型构成了函数重载

为了支持对自定义类型Date类的流输入和流输出，我们需要分别为 `istream` 和 `ostream` 类重载针对Date类的 `>>` 和 `<<` 运算符重载

```cpp
class Date {
	// 友元函数 -- 这个函数内部可以使用Date对象访问私有保护成员
	friend ostream& operator<<(ostream& out, const Date& d);
	friend istream& operator>>(istream& in, Date& d);
	//...
}
//流输出重载
inline ostream& operator<<(ostream& out, const Date& d) {
    //无法访问私有，使用友元
    //out << _year << "-" << _month << "-" << _day << endl;
	out << d._year << "-" << d._month << "-" << d._day << endl;
	return out;
}
//流提取重载
inline istream& operator>>(istream& in, Date& d) {
	in >> d._year >> d._month >> d._day;
	assert(d.CheckDate());
	return in;
}
```

实习的细节

* 因为是针对`istream` 和 `ostream` 类运算符重载，所以第一个参数必须是 `istream& in` 和 `ostream& out`，所以这个重载不能写在Date类里面，因为在Date里的话默认第一个参数是this指针，因此一定要在Date类外实现
* 在Date类外实习的话又会产生取不到私有的 `_year` 等Date类成员的问题
* 所以最后的解决方案是将重载函数设置为Date类的友元，以便让其取到Date类成员

## *取地址及const取地址操作符重载*

* 它们是默认成员函数，我们不写编译器也会自动生成。自动生成就够用了，所以一般是不需要我们自己写的
* 非常特殊的场景：不想让别人取到这个类型对象的地址（或者设置为私有直接报错）

```cpp
Date* Date::operator&() {
    return nullptr;
}
const Date* Date::operator&() const {
    return nullptr;
}
```

# 类与对象（下）

## *构造函数的问题及其解决方法*

### 构造函数体赋值/函数体内初始化

```cpp
class Time {
public:
    Time(int hour = 0) {// 全缺省的默认构造函数，否则会报“没有可用的默认构造函数”的错
        _hour = hour;
    }
private:
    int _hour;
};
class Date() {
public:
    Date(int year, int hour) {
        _year = year;
        Time t(hour);
        _t = t;
        // 自定义结构Time成员hour是私有的，无法直接被Date取到
        // 只能通过在Date的构造函数中新建一个Time类后再赋值给Date的属性_t
    }
private:
    int _year;
    Time _t;
};
int main() {
    Date d(2022, 1);
    return 0;
}
```

* 构造函数体中的语句只能将其称为赋初值，而不能称作初始化，因为初始化只能有一次，而构造函数体内可以多次赋值
* 若类成员中包含本身没有默认构造函数的自定义结构时初始化会很麻烦，如上所示，自定义结构Time成员hour是私有的，无法直接被Date取到，只能通过在Date的构造函数中新建一个Time类后再赋值给Date的属性_t（前提是自定义类有自己的默认构造函数）
* 通过上述方式初始化起始本质也是通过初始化列表初始化的，相当于绕了一个大圈子

### 初始化列表 Initializaition list

* 示例

    ```cpp
    class Time {
    public:
        Time(int hour = 0) {// 全缺省的默认构造函数，否则会报“没有可用的默认构造函数”的错
                _hour = hour;
        }
    private:
        int _hour;
    };
    class Date
    {
    public:
        // 初始化列表可以认为是类成员定义的地方
        Date(int year, int hour, int& ref)
            :_year(year)
            , _t(hour)
            , _ref(ref)
            , _n(10)
        {}
    private:
        // 声明
        int _year = 0; // 内置对象给缺省值，该缺省值是给初始化列表的，此时内置类型若没有显式给值就会用这个缺省值
        Time _t; // 自定义类型成员
        int& _ref; // 引用成员变量
        const int _n; // const成员变量
    };
    
    int main() {
        int y = 0;
        Date d(2022, 1, y); // 对象整体定义
    }
    ```
    
* **初始化列表可以认为是成员变量初始化的地方**。初始化列表是被自动调用的，即使是完全空的构造函数也会自动调用，因此可以认为构造函数里的是二次赋值，private里的只是声明，真正的初始化是在初始化列表之中。也就是说**实例化的过程是：类属性声明（若有缺省值就直接初始化）`->` 初始化列表赋值 `->` 构造函数内部赋值（若有的话） **
* 注意点
  * 每个成员变量在初始化列表中只能出现一次，因为初始化只能有一次
  * 当类中包含以下成员时，其**必须**放在初始化列表位置进行初始化
    * 自定义类型成员，且该类没有默认构造函数时，否则会编译错误
    * 引用成员变量，引用在定义时必须初始化
    * const成员变量，const只有一次定义机会，之后不能重新赋值
  * 推荐使用初始化列表进行初始化
    * 有默认构造函数的自定义类型也推荐用初始化列表。通过调试可以发现，有默认构造函数的自定义结构通过构造函数初始化时也是借助构造函数，然后再赋值。不如直接使用列表初始化，只需要调用一次构造函数即可。
    * 内置类型也推荐使用初始化列表，当然内置类型在函数体内初始化也没有明显的问题
  * 统一的建议：能使用初始化列表就使用初始化列表来初始化，基本不会有什么问题，肯定比在函数体内好
* 尽量使用初始化列表初始化
* 成员变量在类中的声明次序就是其在初始化列表中的初始化顺序，与其在初始化列表中的先后次序无关

    <img src="声明顺序与初始化顺序的俄关系.png" width="60%">

    可以发现上面程序的结果是1和随机值。这是因为根据声明，`_a2` 在初始化列表中应该首先被定义，其定义值为 `_a1`，然而此时 `_a1` 还没有被定义，所以其为随机值

### `explicit` 关键字：禁止单参数类的隐式类型转换，常用于string类

这部分之后重新看一下8.05的1:07:00左右

**内置类型是可以隐式类型转换成自定义类型的**，编译器会自动调构造和拷贝构造（有些编译器自动优化为只有自定义）

可以通过 `explicit` 关键字来禁止这种**单参数**的隐式类型转换

```cpp
class Date {
public:
    // 1. 单参构造函数，没有使用explicit修饰，具有类型转换作用
    // explicit修饰构造函数，禁止类型转换---explicit去掉之后，代码可以通过编译
    explicit Date(int year)
            :_year(year)
    {}
    /*
    // 2. 虽然有多个参数，但是创建对象时后两个参数可以不传递，没有使用explicit修饰，具有类型转换作用
    // explicit修饰构造函数，禁止类型转换
    explicit Date(int year, int month = 1, int day = 1)
    : _year(year)
    , _month(month)
    , _day(day)
    {}
    */
    Date& operator=(const Date& d) {
        if (this != &d){
            _year = d._year;
            _month = d._month;
            _day = d._day;
        }
        return *this;
    }
private:
    int _year;
    int _month;
    int _day;
};
```

`Date d1(2022);` 和 `Date d2 = 2022;` 虽然结果是一样的，但过程并不一样，前者是直接调用构造函数，而后者包含了先构造（创建临时变量）和拷贝构造（隐式类型转换）以及编译器的优化

### 委托构造

### 匿名对象 Anonymous object 及其单参数构造函数

`Date(2000);` 生命周期只有这一行

可以在有仿函数参数的函数模板中使用

## *static成员*

### 概念

**有些时候类需要它的一些成员与类本身直接相关，而不是与类的各个对象保持关联**。比方说银行账户类会规定基准利率，这个基准利率是对每个账户对象一致成立的，所以它属于这个类的共性，因此没有必要让每个类对象都存储它。当修改它的时候应该要对所有的类对象都成立

静态成员就是专门给这个类访问，**静态成员变量一定要在类外进行初始化**，注意在类外定义的时候必须要用 `::` 指定类域，否则编译器找不到

```cpp
class A {
public:
    A() { ++_scount; }
    A(const A& t) { ++_scount; }
    ~A() { --_scount; }
    static int GetACount() { return _scount; } // 静态成员函数，没有this指针，不能访问非静态成员
private:
    static int _scount; // 静态成员变量声明
};

// 在类外面定义初始化静态成员变量
int A::_scount = 0;

int main() {
    A a1;
    A a2;
    cout << a1.GetACount() << endl; // 用静态成员函数来取
}
```

### 特性

* 静态成员为所有类对象所共享，属于整个类，**也属于这个类的所有对象，并不单独属于某个具体的对象，这点在 `std::shared_ptr` 的设计中是一个坑**，存放在静态区
* 和友元一样，静态成员变量**必须在类外定义和初始化**（但是静态成员函数可以在类内定义），定义时不添加static关键字，类中只是声明，如 `int A::scount = 0;`，若是模板的话要把模板也给带上。初始化列表不能用来初始化静态成员
* 静态成员变量也受访问限定符的限制
* 当static是公有的时候，类静态成员可用 `类名::静态成员` 或者 `对象.静态成员` 来访问。但是若私有，则只能通过 `类名::静态成员` 访问或静态成员函数来获取，这是因为类静态成员不属于某个类对象，而是存放于静态区中，属于类域
* **静态成员函数没有隐藏的this指针，不能访问任何非静态成员**，也不能声明const成员函数，也不能在static函数体内使用this指针
* 静态变量不能给缺省值，只能在类外面给初始值，但 `const int` 类型的静态变量可以给缺省值，比如哈希桶中的素数size扩容就用到了这个特性

### 一个应用：设计一个只能在栈上定义对象的类

若把构造函数定义为public，那么类对象可以定义在任何地方。但若把构造函数定义为private，然后给一个静态函数，里面定义类后返回

静态成员函数没有this指针，不需要访问对象来调用。若不用static，就会产生调用成员函数需要现创建对象的矛盾，因为调用成员函数需要this指针指向对象

```cpp
class StackOnly{
public:
    static StackOnly CreateObj(){
        StackOnly so;
        return so;
    }
private:
    StackOnly(int x = 0, int y = 0)
        :_x(x)
        , _y(0)
    {}
    int _x = 0;
    int _y = 0;
};
```

### 调用问题

* 静态成员函数可以调用非静态成员函数吗？不能，因为静态成员函数没有this指针
* 非静态函数可以调用类的静态成员函数吗？可以，因为静态成员属于整个类和类的所有对象

## *友元 Friend*

有一些函数虽然不属于类，但类也要用到他们并且允许他们访问类的非公有成员从而实现某些功能。友元提供了一种突破封装的方式，在一些使用场景下提供了便利。但是友元会增加耦合度，破坏封装，所以要尽可能少地使用友元

### 友元函数

友元函数可以直接访问类的私有成员，它是定义在类外部的普通函数，不属于任何类，但需要在类的内部声明，声明时需要加 `friend` 关键字

注意：友元在类内部的声明不是普通的声明，相当于只是制定了访问的权限，还需要在头文件的**类外部再次声明**然后在c文件中定义

* 运用场景
  * 重载运算符：`operator<<` 、`operator>>` 当定义为类成员函数时，cout的输出流对象在和隐含的this指针抢占第一个参数的位置。为了保证第一个形参为cout，需要将改运算符重载定义为全局函数，但此时又会导致它是类外成员而无法取得类属性成员，因此要借助友元函数帮助
  * 多个类之间共享数据，如有一个函数 `void func(const A& a, const B& b, const C& c)`，设置func同时成为A、B、C类的友元函数以获取多个类之中的数据
* 说明
  * 友元函数可以访问类的所有成员，包括私有保护成员，但不是类的成员函数，所有它没有this指针
  * 友元函数不能用 `const` 修饰，因为在类中 `const` 修饰的是this指针
  * 友元函数可以在类定义的任何地方声明，不受类访问限定符限制
  * 一个函数可以是多个类的友元函数
  * 友元函数的调用与普通函数的调用原理相同

### 友元类

* 友元类的所有成员函数都可以是另一个类的成员函数，都可以访问另一个类中的非公有成员
* 特性
  * 友元关系是单向的，不具有交换性
  * 友元关系不能传递
  * 友元关系不能继承

## *内部类 Inner Class（Java常用）*

### 概念

如果一个类定义在另一个类的内部，这个内部类就叫做内部类。内部类是一个独立的类，它不属于外部类，更不能通过外部类的对象去访问内部类的成员。外部类对内部类没有任何优越的访问权限

### 特性

* 内部类受外部类的类域限制，也受访问限定符的限制
* 内部类天生是外部类的友元
* sizeof(外部类)==外部类，和内部类没有关系

## *特殊类设计*

### 不能被拷贝的类

* C++98

  ```cpp
  class CopyBan {
      // ...
  private:
      CopyBan(const CopyBan&);
      CopyBan& operator=(const CopyBan&);
      //...
  };
  ```

  * 将拷贝构造函数与赋值运算符重载只声明不定义
  * 将拷贝构造和赋值运算符重载定义为私有，这样就不能在类外面实现了
  * 如果非要说缺点的话，那就是还有一种情况，可以在类里重新定义拷贝构造

* C++11：使用 `delete` 关键字禁止编译器默认生成拷贝构造和复制运算符重载

  ```cpp
  class CopyBan {
      // ...
      CopyBan(const CopyBan&)=delete;
      CopyBan& operator=(const CopyBan&)=delete;
      //...
  };
  ```

* 防拷贝应用
  * `unique_ptr`
  * thread
  * mutex
  * istream
  * ostream

### 只能在堆上创建对象的类

创建类有三种构造方法+1种拷贝方法

```cpp
HeapOnly obj1; //栈上
static HeapOnly obj2; //静态区
HeapOnly* obj3 = new HeapOnly; //堆上
HeapOnly obj4(obj1); //拷贝
```

要创建对象需要通过构造函数和拷贝构造，只要想办法把它们堵死到只能在堆上创建对象就可以

* 将类的构造函数私有，拷贝构造声明成私有，防止别人调用拷贝在栈上生成对象

* 提供一个静态的成员函数，在该静态成员函数中完成堆对象的创建

  * 调用成员函数 `CreateObject` 需要现有类对象，然后类对象又只能通过成员函数得到，一个经典的先有鸡还是先有蛋问题
  * 因此将 `CreateObject` 设置为不需要没有this指针的静态成员函数，且构造函数不需要this指针就可以调用

  ```cpp
  class HeapOnly {
  public:
      static HeapOnly* CreateObject() {
          return new HeapOnly;
      }
  private:
      HeapOnly() {}
      HeapOnly(const HeapOnly&) = delete; //防拷贝
  };
  ```

### 只能在栈上创建对象的类

传值返回必然要调用拷贝构造，因为有编译器优化，把传值返回的拷贝构造省去了，相当于直接构造

要new对象必须要调用器其构造函数，设置成私有后new就不能访问构造函数了

这里也不是移动构造，因为默认移动构造必须要不写构造函数

第二种思路是可以重载 `operator new()` 和 `operator delete()`，使无法从堆上创建对象。但这种写法有缺陷，无法阻止全局new对象和静态对象生成

```cpp
class StackOnly {
public:
    static StackOnly* CreateObject() {
        return StackOnly(); //编译器优化，传值返回没有拷贝构造
    }
    StackOnly(const StackOnly&) = delete; //防拷贝
    
    //void* operator new(size_t size) = delete;
	//void operator delete(void* p) = delete;
private:
    StackOnly() {}
};
```

### 不能被继承的类

* C++98：父类构造函数私有化，这样子类就不能调用父类的构造函数，则无法继承
* C++11：用 `final` 关键字修饰父类，也就称为所谓的最终类

```cpp
//C++98
class NonInherit {
public:
	static NonInherit GetInstance() {
		return NonInherit();
	}
private:
	NonInherit() {}
};
//C++11
class A final {};
```

### 只能创建一个对象的类（单例模式 Singleton）

[快速记忆23种设计模式 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/128145128)

设计模式 Design Pattern：是一套被反复使用、多数人知晓的、经过分类的代码设计经验总结，具体可以看上面的文章。比较重要的有适配器模式、单例模式、迭代器模式、观察者模式和工厂模式

单例模式 Singleton Pattern：一个类只能创建一个对象。该模式可以保证系统中在当前进程中该类只有一个实例，并提供一个访问它的全局访问点，该实例被所有程序模块共享。比如在某个服务器程序中，该服务器的配置信息存放在一个文件中，这些配置数据由一个单例对象统一读取，然后服务进程中的其他对象再通过这个单例对象获取这些配置信息

单例模式在何时创建这个唯一的对象有区别，即

* 饿汉模式 Eager Initialization：不管将来用不用，程序启动时就创建一个唯一的实例对象

  ```cpp
  class Singleton {
  public:
  	static Singleton* GetInstance() {
  		return _spInst;
  	}
  	void Print() {
  		cout << _spInst << endl;
  	}
  private:
  	Singleton() {}
  	Singleton(const Singleton&) = delete; //防拷贝
  
  	int _a;
  	//static Singleton _sInst; //声明，否则构造私有取不到，声明的为类成员就能令定义时取到
  	static Singleton* _spInst; //声明
  };
  
  //Singleton Singleton::_sInst; // 定义成全局的，而不是私有的，满足饿汉模式的要求（main函数之前就被初始化）
  Singleton* Singleton::_spInst = new Singleton; //定义
  
  int main() {
  	Singleton::GetInstance()->Print(); //只能这么取，因为不能实例化
  	return 0;
  }
  ```

* 懒汉模式 Lazy Initialization：用到的时候在初始化创s建（延迟加载）。若单例对象构造十分耗时或者会占用很多资源，而有可能这个对象程序运行时并不会用到。若此时在程序时也要对这个对象进行初始化就会令程序启动非常缓慢

  ```cpp
  static Singleton* GetInstance() {
      if (_spInst == nullptr) {  //第一次调用的时候创建对象
          _spInst = new Singleton;
      }
  	return _spInst;
  }
  Singleton* Singleton::_spInst = nullptr; //定义
  ```

实际中**懒汉比较常用**，因为饿汉有时候初始化速度太慢，并且当多个单例存在顺序依赖问题时，饿汉的初始化顺序不确定，而懒汉可以控制初始化顺序

单例不用析构，但是不属于内存泄漏，因为这个类一直在使用。不过可以在析构函数里写信息到文件中持久化，可以通过一个垃圾回收的内部类来实现

```cpp
// 实现一个内嵌垃圾回收类
class CGarbo {
public:
    ~CGarbo(){
    if (Singleton::_spInst)
    	delete Singleton::_spInst;
    }
};
// 定义一个静态成员变量，程序结束时，系统会自动调用它的析构函数从而释放单例对象
static CGarbo Garbo;
```

# 复杂的const问题

因为函数的传参行为和const与变量、引用和指针的初始化是一样的，所以直接用函数传参的例子来总结const

```cpp
// const与变量没有权限关系
// 因为变量都是拷贝，没有内存关系
void test_v1(const int i) { printf("%d\n", i); }
void test_v2(int ci) { printf("%d\n", ci); }

// const与引用有权限放大问题
// 只能是int或const int 赋值给 const int&，不能const int赋值给int&
void test_r1(const int &ri) { printf("%d\n", ri); }
void test_r2(int &ri) { printf("%d\n", ri); }

// const与指针有权限放大问题
// 只能是int*或const int* 赋值给 const int*，不能const int*赋值给int*
void test_p1(const int *pi) { printf("%d\n", *pi); }
void test_p2(int *pi) { printf("%d\n", *pi); }

int main()
{
    int i = 1;
    test_v1(i);

    const int ci = 2;
    test_v2(ci);

    int *pi = &i;
    const int *cpi2 = &i; //用const int*指向int* 权限缩小ok
    const int *cpi = &ci;
    //int *pci = &ci; //用int*指向const int* 权限放大了报错

    test_r1(i);
    test_r1(ci);
    test_r2(i);
    //test_r2(ci); //权限放大

    test_p1(pi);
    test_p1(cpi);
    test_p2(pi);
    //test_p2(cpi); //权限放大
    return 0;
}
```

## *const与变量*

const的核心思想是让一个对象不可被修改，一旦尝试修改就报错

```c
const int buffSize = 512; // 规定缓冲区的大小
buffSize = 1024; // 对const赋值，会直接报错
```

### 修饰变量：置入常量区

* 局部变量：一旦写定就不能再更改了，所以const对象必须初始化。通常被用来修饰只读的数组和字符串
* 全局变量：这种写法应当尽量避免，很容易造成定义冲突
* 编译器通常不为普通const常量新开存储空间，而是将它们保存在符号表中，这使得它成为一个编译期间的常量，没有了存储与读内存的操作，使得它的效率也很高

### const与初始化

因为const对象一旦创建后它的值就不能再被更改，所以**const对象必须要被初始化**

**利用const对象去初始化另外一个对象不会对被初始化的对象造成任何影响**，这是因为const对象处于 `.rodata` 段（所谓的常量区域），而新的变量可能在任意位置（`.rodata`、`.data`、`.text`），初始化只是将const对象的值放到寄存器里，然后再mov给新的对象。一旦拷贝完成，const对象和被初始化的新对象也就没有半毛钱关系了

### const的底层

对于全局const常量，比如 `const char* ch = "abcdef";` 都会放到 `.rodata`（常量区）只读段。这个段属性（标识位） `sh_flag` 为 `SHF_ALLOC`，没有 `SHF_WRITE`，所以不允许修改它（自我修养P79）

全局const常量会将 `ch` 加入符号表中，因此在其他文件中使用它也需要extern声明，和普通变量的定义和声明是一样的

而局部const变量，则不会放入符号表中，在栈帧中放到某个页中，通过将页属性设置为只读就可以保证局部变量为只读

## *const与引用*

### 常量引用 Reference to const 与权限放大问题

引用对象是const，即对常量的引用或者叫常量引用：对常量的引用不能被用作修改它所绑定的对象，注意：**常量引用本身必须是任意类型的const**，因为若引用本身不是const，就会有权限放大问题

```cpp
int a = 10;
int& b = a; // 权限平移
cout << typeid(a).name() << endl;
cout << typeid(b).name() << endl;

const int c = 20;
cout << typeid(c).name() << endl;
// int& d = c; // 权限放大，这么写是错误
const int d = c;

int e = 30;
const int& f = e; // 权限缩小，即e可以通过其他方式改变，但f不能改变e

int ii = 1;
double dd = ii; // 隐式类型转换
```

* 权限不能放大，但可以缩小，也就是说**可以用常量来引用非常量，但反过来不行**
* const引用具有很强的接受度，因为其权限比较小
* 若使用引用传参且函数内若不改变传入参数，建议尽量用const传参

### 例外：常量引用的类型不必严格匹配

```cpp
// double dd = (double)ii;
// cout << typeid(ii).name() << endl;
// 强制转换只是创建了一个double的临时变量，不改变原变量ii

// 和整体提升过程一样 在转换过程中产生的是具有常性的 const double 临时变量
// double& rdd = ii; // 因此这么写是错误的，本质原因是权限缩小
const double& rdd = ii;

const int& x = 10; // 甚至可以引用常量
```

初始化常量引用时允许用任意表达式作为初始值，只要该表达式的结果能隐式转换成功

## *const与指针*

### 常量指针与指针常量

* const修饰指针的三种情况

  * 常量指针 Pointer to const/底层const low-level const：`int const *p` 或者 `const int *p`：cosnt修饰的是 `*p`，不能修改 `*p`所指向的值，但可以修改p地址本身

  * 指针常量 Const pointer/顶层const top-level const：`int* const p`：p一个指针常量，不能修改指针p，即p指向的地址不可修改，但 `*p` 指向的地址所存储的值可以修改

  * `const int* const p`：同时是指针常量和常量指针

  * 快速的记忆方法

    从右到左，遇到p就翻译成p is a，遇到 `*` 就翻译成 point to，比如`int const *p` 是 p is a point(er) to int const；`int* const p` 是 p is const point to int 。如何理解常量指针与指针常量？ - 李鹏的回答 - 知乎 https://www.zhihu.com/question/19829354/answer/44950608

    一个永远不会忘记的方法，**const默认是修饰它左边的符号的，如果左边没有，那么就修饰它右边的符号**

    1.  `const int *p` 左边没有，看右边的一个，是int，自然就是p指针指向的值不能改变

    2. `int const *p` 此时左边有int，其实和上面一样，还是修饰的int
    3. `int* const p` 修饰的是*，指针不能改变
    4. `const int *const p` 第一个左边没有，所以修饰的是右边的int，第二个左边有，所以修饰的是 * ，因此指针和指针指向的值都不能改变
    5. `const int const *p` 这里两个修饰的都是int了，所以重复修饰了，有的编译器可以通过，但是会有警告，你重复修饰了，有的可能直接编译不过去

* 只允许权限缩小或平移，不允许权限放大。简单的说就是 `int*`和 `const int*` 都可以赋值给 `const int*`，但是 `const int*` 不能赋值给 `int*`

  * 当对常变量取地址时，必须要**将指针指向的量设为const**（常量指针 `int const *p`），否则会有权限放大问题；将指向const的指针常量用const数据或非const数据的地址（权限缩小）初始化为或赋值是合法的。

    ```c
    const int arr[] = {0};
    const int *ptr = arr; // 原来是const的数据，其指针指向的量也必须设置为const
    ```

    注意这里不要混淆了，不能用const数据初始化指针常量，属于权限放大

  * 但是只能把非const的指针赋给普通数据，否则会权限放大
  
    ```c
    const int arr[3] = { 0,1,2 };
    int *ptr = arr; // 编译器不会报错，但ptr权限被放大了，可以通过*ptr修改arr中的数据，这种写法仍然是错误的
    ptr[2] = 3
    ```

### constexpr变量

在一个复杂系统中是很难确定到底一个

### 指针、常量和类型别名

## *const与函数*

函数传参时等同于变量赋值，所以它的规则和前面const与变量、引用、指针的关系一模一样

### const传参和返回的作用

* 修饰函数参数

  * 防止修改指针指向的内容
  * 防止修改指针指向的地址

* 修饰函数返回值

  若以传指针返回加const修饰，那么函数返回的内容不能被修改，且该返回值只能以加const修饰的同类型指针接收

  ```C
  const char* GetString(void);
  char* str = GetString(); // 错误
  const char* str = GetString(); // 正确
  ```

### Tips：尽量使用常量引用做形参

const类型的左值可以用来接受const或者非const，所以比较保险；但反过来非const就无法接受const了，因为有权限放大

### const与函数重载

## *const与类*

在[const成员函数](#const成员函数)里有说明了

# C++内存管理 Memory Management

## *C/C++内存分布（详见操作系统）*

<img src="addressSpace.png" width="70%">

* 堆会多申请一些空间来存放和堆自身有关的属性信息，即cookie数据
* static修饰局部变量的本质就是将该变量开辟在全局区域
* 启动可执行程序，也就是系统新建进程时会换入除堆和栈之外的代码，堆和栈只有在真正要用的时候才会开始开辟内存

### 一道经典例题

```cpp
int globalVar = 1; //globalVar存在数据段（静态区）
static int staticGlobalVar = 1; //staticGlobalVar存在数据段（静态区）
void Test() {
    static int staticVar = 1; //staticVar存在数据段（静态区）
    int localVar = 1; //localVar存在栈上
    int num1[10] = { 1, 2, 3, 4 }; //num1存在栈上
    char char2[] = "abcd"; //这个表达式的意义是将常量区的"abcd\0"拷贝一份放到栈上给char2数组，char2存在栈上，*char2指向首元素也在栈上
    const char* pChar3 = "abcd"; //直接指向常量区的字符串
    int* ptr1 = (int*)malloc(sizeof(int) * 4);
    int* ptr2 = (int*)calloc(4, sizeof(int));
    int* ptr3 = (int*)realloc(ptr2, sizeof(int) * 4);
    free(ptr1);
    free(ptr3);
}
```

这道题最容易出错的地方在于 `char2` 和 `pChar3` ，如下图`“abcd\0”` 是位于常量区的字符串，前者的意义是把该字符串拷贝一份给char2数组，后足的意义是 `pChar3` 这个指针指向常量区的字符串

<img src="内存区经典例题.png" width="20%">

## *C++中的动态内存管理*

### 基本操作

```cpp
int *p1 = (int*)malloc(sizeof(int));
int *p2 = new int;
int *p3 = new int[5]; //申请5个int的数组
int *p4 = new int(5); //申请1个int对象，初始化为5
//C++11支持用初始化列表初始化数组
int *p5 = new int[5]{1,2,3,4,5};

delete p2;
delete[] p3; //括号要匹配
```

* `new/delete` 不是函数，而是操作符，C++没有和calloc和realloc对应的操作符

* new/delete操作内置类型：new/delete跟malloc/free没有本质的区别，只有用法的区别，new/delete简化了一些和注意操作符匹配

* new/delete操作自定义类型：**new/delete 是为自定义类型准备的。不仅在堆上申请出来，还会调用构造函数初始化和析构函数清理**。若使用malloc那么即使是对自定义类型的初始化都很难，比如类成员是私有的不提供接口获取怎么办呢？

* 注意 `new/delete` 和 `new[]/delete[]` 匹配使用，否则可能就会出问题

* `malloc` 失败返回 `NULL`，每次都要检查返回值防止野指针；`new` 失败时抛异常 `bad_alloc`，不需要检查返回值

  ```cpp
  try {
      char *p = new char[100];
  }
  catch (const exception& e) {
      // ...
  }
  ```

### `[]` 不匹配错误的底层原因

注意：对于内存的处理与编译器类型有很大的关系，下面的情况仅针对VS系列

内置类型因为没有构造和析构函数，所以影响没有那么大，但对于自定义类型影响就大了

由于语法设计问题，`new T[5]` 给了要new的元素个数，这告诉编译器要对于T类元素调用1次malloc+5次构造，但是 `delete[]` 就没有告诉编译器到底应该要调用多少次析构+1次free

<img src="delete指针偏移问题.png" width="50%">

为了解决这个问题，VS系列采取的方法是在多new一个int的空间，即在new出来的空间头上记录有几个元素，这就产生了指针偏移问题，因为返回的指针实际上是真正的头指针+4

因此若是 `delete[]` 就会自动进行指针偏移，此时如果不匹配的使用 `delete` 就会取到错误的头指针，因为实际的头指针为 `(char*)ptr-4` ，编译器就会报错

## *空间配置器 Allocator*

### 空间配置器介绍

<img src="内存池结构.png">

STL容器需要频繁申请释放内存，若每次都要调用malloc进行申请会有以下缺点

* 空间申请与释放需要用户自己管理，容易造成内存泄漏
* 频繁向系统申请内存块，容易造成内存碎片也影响程序运行效率
  * 外碎片问题：频繁向系统申请小块内存，有足够内存但是不连续，无法申请大块内存
  * 内碎片问题：内存块挂起来管理，由于按照一定规则对齐，就会导致内存池中的内碎片
* 直接使用malloc与new进行申请，每块空间有额外空间浪费，因为要记录开的内存空间的相关信息，如大小等
* 代码结构不清晰
* 未考虑线程安全

因此需要设计高效的内存管理机制，空间配置并不是什么特殊的数据结构，只是对malloc的深度封装以提高它的效率

### 一级空间配置器

```mermaid
flowchart TB
id1(二级空间配置器 __dafault_alloc_template)-->choose1{大于128吗}
choose1--Yes-->id2(交由一级空间配置器 __malloc_alloc_template 申请)
choose1--No-->id3(确认到哪个桶中去找)
id3-->choose2{该桶下是否挂内存块}
choose2--Yes-->id4(直接将第一个小块内存返回给用户)
choose2--No-->id5(向该桶中填充小块内存并返回给用户)
```

一级空间配置器是二级空间配置器当申请空间大于128BByte时的特例，而一级空间配置器就是直接用 `__malloc_alloc_template` 封装了malloc和free，二层空间配置器的底层则是对malloc和free的多次封装

### 内存池 Memory Pool

<img src="内存池.png">

内存池 Memory Pool 就是先申请一块比较大的内存块作为备用。当需要内存时，直接取内存池中去取。首位两个指针相当于是限制这个蓄水池的闸门。若池中空间不够了，就继续向内存中malloc取。若用户不使用了就可以直接交还给内存中，内存池可以位之后需要的用户重新派发这些回收的内存

那么这块内存池该采取怎么样的数据结构进行管理呢？考虑到主要问题在于归还内存的时候可能并不是按照给的时候的顺序给的，即使归还了之后也要重新将内存分派出去，那么如何知道内存块的大小呢？若将内存池设计成链表的话，需要挨个遍历效率低。因此二级空间配置器设计成了哈希桶的形式

注意：内存池和哈希表实现的二级空间配置器是两个结构，空间配置器是一个管理结构，它会去找内存池要，内存池可以通过malloc向内存要资源从而不断扩展，相当于 `end_free` 这个闸门一直在外后走。在同一个进程中所有的容器共享一个空间配置器，==因此STL空间配置器是用单例模式 Singleton Pattern设计的==

### 二级空间配置器设计

<img src="二级空间配置器的哈希桶设计.png">

因为已经是小于128字节的内存才会使用二级空间配置器，因此如果继续用1字节进行切割的话那么小内存就太碎了，而且用户申请的时候的内存基本以4字节的倍数为主，其他大小的空间几乎很少用到，因此STL中的设计是将内存对齐到8字节，也就是说128个字节内存有16个桶

### SGI-STL二级空间配置的空间申请与回收

* 前期准备

  用联合体来维护哈希桶结构：这块内存空间是两用的，申请了之后所有的都可以用，但如果只是挂在哈希桶上，那么需要用头4个或头8个字节来存下一个结点的地址

  ```cpp
  union obj {
      union obj * free_list_link;
      char client_data[1]; /* The client sees this. */
  };
  ```

* 申请空间

  * 向内存池中索要空间：哈希桶为空？去内存池中要。申请一个结点会给20个结点，返回头上那个给用户，然后将多余的19个相同字节切割的内存块全部挂到对应的哈希桶上
  * 向哈希桶中填充内存块：内存池为空？去找以及空间配置器补充

* 空间回收

  ```cpp
  // 函数功能：用户将空间归还给空间配置器
  // 参数：p空间首地址 n空间总大小
  static void deallocate(void *p, size_t n) {
      obj *q = (obj *)p;
      obj ** my_free_list;
  	// 如果空间不是小块内存，交给一级空间配置器回收
      if (n > (size_t) __MAX_BYTES) {
          malloc_alloc::deallocate(p, n);
          return;
      }
      // 找到对应的哈希桶，将内存挂在哈希桶中（头插）
      my_free_list = free_list + FREELIST_INDEX(n); //找到对应桶的序号
      q->free_list_link = *my_free_list; //头插
      *my_free_list = q;
  }
  ```

### 空间配置器的默认选择

SGI-STL默认选择使用一级还是二级空间配置器，通过 `USE_MALLOC` 宏进行控制

### 空间配置器的二次封装

下面是源码：以stl_list容器为例，STL中的容器都遵循类似的封装步骤

<img src="STL容器对空间配置器的封装.png">

## *new与delete操作符的底层原理*

### 实现原理

* new操作符 = `operator new()` 全局函数申请空间 + 调构造函数
* `new T[N]` =`operator new[]()` 全局函数申请N个对象空间 + 调N次构造函数

* delete操作符 = 调析构函数清理对象中的资源 + `operator delete` 全局函数
* `delete T[N]` = 调N次析构函数清理N个对象空间中的资源 + N次 `operator delete[]()` 全局函数回收空间

### `operator new()` 与 `operator delete()` 全局函数

`operator new()` 与 `operator delete()` 是系统提供的全局函数，`new` 在底层调用 `operator new()` 全局函数（注意不是运算符重载）来申请空间， `delete` 在底层通过 `operator delete()` 全局函数来释放空间

* `operator new()` 源代码

    ```cpp
    void *__CRTDECL operator new(size_t size) _THROW1(_STD bad_alloc) {
        // try to allocate size bytes
        void *p;
        while ((p = malloc(size)) == 0) {
            if (_callnewh(size) == 0) {
                // report no memory
                // 如果申请内存失败了，这里会抛出bad_alloc 类型异常
                static const std::bad_alloc nomem;
                _RAISE(nomem);
            }
        }
        return (p);
    }
    ```

    * 该函数实际通过malloc来申请空间，当malloc申请空间成功时直接返回
    * 封装malloc的原因是若申请空间失败，符合C++ new的失败机制：抛异常
    * 可以直接使用这个全局函数 `char *p = (char*)operator new(100);`，当然实际中是不会这么用的

* `operator delete()` 源代码

    ```cpp
    #define free(p) _free_dbg(p, _NORMAL_BLOCK) //free的实现，封装free宏
    void operator delete(void *pUserData) {
        _CrtMemBlockHeader * pHead;
        RTCCALLBACK(_RTC_Free_hook, (pUserData, 0));
        if (pUserData == NULL)
            return;
        _mlock(_HEAP_LOCK); /* block other threads */
        __TRY
            /* get a pointer to memory block header */
            pHead = pHdr(pUserData);
            /* verify block type */
            _ASSERTE(_BLOCK_TYPE_IS_VALID(pHead->nBlockUse));
            _free_dbg( pUserData, pHead->nBlockUse ); //free的实现
        __FINALLY
            _munlock(_HEAP_LOCK); /* release other threads */
        __END_TRY_FINALLY
        return;
    }
    ```


​		`operator delete()`：该函数最终是通过free来释放空间的。实际上是可以直接free的，但为了和 `operator new()` 凑一对，又进行了一次封装

### 重载 `operator new()` 与 `operator delete()`

一般不需要重载，除非在申请和释放空间的时候有某些特殊的需求，比如

* 打印日志信息

* 重载一个类专属的 `operator new()`：利用空间配置器，当然实际STL的源码封装和下面的思路是不同的，可以看上面的封装

  ```cpp
  struct ListNode {
      int _val;
      ListNode* _next;
      static allocator<ListNode> alloc; //所有类对象共享一个空间配置器
      void* operator new(size_t n) { //重载类专属的operator new()
          void* obj = _alloc.allocate(1);
          return obj;
      }
      
      void operator delete(void* ptr) {
          _alloc.deallocate(ptr, 1);
      }
  }
  ```

### 定位new表达式（placement-new）

使用格式：`new(place_address)type` 或 `new(place_address)type(initializer-list)` 

定位new表达式是在已经分配好的内存空间中调用构造函数初始化一个对象，因为如之前所述用malloc系列函数是取不到私有类成员的

使用场景：因为new出来的已经调用构造初始化过了，所以是专门给malloc用的。一般都会配合内存池进行使用，因为内存池都是通过malloc向系统系统申请的，因此内存池分配出的内存是没有初始化过的，所以若是自定义类型对象，需要使用定位new

```cpp
class A {};
A *p1 = (A*)malloc(sizeof(A));
if (p1 == nullptr) {
    perror("malloc fail");
}
new(p1)A(10); //初始化为10
```

## *`malloc/free` 和 `new/delete` 的区别*

语法使用的区别

本质功能的区别：申请自定义类型对象时

## *内存泄漏*

### 什么是内存泄漏 memory leak

在main函数退出后，所有的栈、堆空间都会被OS回收，所以内存泄漏不是指分配的内存没了。而是指当进程长期运行时，比如服务器、应用软件等**失去了对分配内存的控制**，也就是丢失了某块内存的指针

特别是在某一个循环中不断泄漏内存后，因为OS会认为这块内容一直都在被用户使用，所以就会造成OS的可用内存越来越少，最终造成进程甚至是整台主机的宕机

### 内存泄漏分类

* 堆内存泄漏：程序的设计错误导致在某个局部调用的函数中通过 `malloc` 或 `new` 得到的动态开辟内存没有被释放，并且丢失了这块内容的指针入口
* 系统资源泄漏：指OS为程序分配的系统资源，如socket或者其他fd、管道等没有使用对应的函数释放掉，导致系统资源的浪费，严重可导致系统效能减少，系统执行不稳定。这种资源泄漏是无法通过关闭进程来重置的，某些时候只能通过重启系统

### 如何检测内存泄漏

检测工具内部原理：申请内存时用一个容器记录下来，释放内存时，从容器中删除掉

`valgrind`

### 如何避免内存泄漏

# 模板 Template 与STL

## *泛型编程 Generic Programming*

泛型编程：编写与类型无关的通用代码，是代码复用的一种手段，模板是泛型编程的基础

## *函数模板*

### 函数模板概念

* 函数模板代表了一个函数家族，该函数模板与类型无关，在使用时被参数化，根据实参类型产生函数的特定类型版本
* 模板不支持分离编译，统一放在头文件里，但在同一个文件里可以分离编译

```cpp
// 模板参数定义了模板类型，类似函数参数，但函数参数定义了参数对象
// typename后面类型名字T是随便取的，一般是大写字母或者单词首字母大写 T、Ty、K、V，代表了一个模拟类型/虚拟类型
template<typename T>
void Swap(T& left, T& right) {
    T temp = left;
    left = right;
    right = temp;
}
// swap在std中有模板定义了
```

### 函数模板的原理：类型推演 Type Deduction和函数模板实例化 Instantiation

函数模板是一个设计图，他本来并不是真正用来执行任务的函数，每次调用相关函数编译器都要根据函数模板进行一次类型推演产生相关函数

### 函数模板的实例化

* 隐式实例化：让编译器根据实参自动推演模板参数的实际类型。模板中不会进行隐式类型转换，编译器不知道该转成哪一个会报错，需要用户自已进行强制类型转换
* 显式实例化：在函数名后的 `<>` 中由用户指定模板参数的实际类型
  * 参数需要强转时可以使用

      ```cpp
      // Add(1.1, 2) // 报错
      Add((int)1.1, 2); // 用户强转输入
      Add<int>(1.1, 2); // 模板的显式实例化
      ```

  * 需要指定返回类型时一定要显式实例化

    ```cpp
    // 已经定义了一个A类
    template<class T>
    T* Func(int n) {
        T* a = new T[n];
        return a;
    }
    // 不能自动推演返回类型T
    Func<A>(10); // 显式实例化
    ```

## *类模板*

### 类模板的定义格式

```cpp
// template<typename T>
template<class T> // 上下两种类模板都可以
class Stack {
    //...
private:
    T* _a;
};
```

### 类模板的实例化

类模板需要显式实例化，类模板名字不是真正的类，实例化的结果 `Stack<int>` 才是真正的类，若不显式实例化，编译器无法判断

## *非类型模板参数 Nontype*

```cpp
template<class T, size_t N = 10>
class array {
public:
private:
    T _a[N];
};
```

* 非类型模板参数只能用于整形，浮点数、类对象以及字符串是不允许作为非类型模板参数的
* 非类型的模板参数必须在编译期就能确认结果

## *类模板的特化 Specialization -- 针对某些类型进行特殊化处理*

### 函数模板特化

```cpp
// 函数模板
template<typename T>
bool Greater(T left, T right) {
    return left > right;
}
// 若输入的参数是指针，就不会进行实例化，而是直接进入下面的特化
template<>
bool Greater<Date*>(Date* left, Date* right) {
    return *left > *right;
}
```

* 必须要现有一个基础的函数模板
* 关键字 `template` 后面接一对空的尖括号 `<>`
* 函数名后跟一对 `<>`，尖括号中指定需要特化的类型
* 必须要使用模板参数关键字 `typename` 或 `class`，这两者在定义模板是等价的，不能将 `class` 换成 `struct`

### 类模板特化

* 正常模板

    ```cpp
    template<class T1, class T2>
    class Data{};
    ```

* 全特化

    ```cpp
    template<>
    class Data<int, char>{};
    ```

* 偏特化
  * 部分特化：将模板参数类表中的一部分参数特化

    ```cpp
    template<class T1>
    class Data<T1, int>{};
    
    //调用
    Data(int, int); // 此时会进入特化，因为第二个参数类型匹配
    ```

  * 参数更进一步的限制

    ```cpp
    template<class T1, class T2>
    class Data<T1*, T2*>{};
    
    template<class T1&, class T2&>
    class Data<T1&, T2&>{};
    ```

## 模板参数的匹配原则

* 一个非模板函数可以和一个同名的函数模板同时存在，而且该函数模板还可以被实例化为这个非模板函数
* 匹配的非模板参数函数/类 > 最匹配、最特化 > 一般模板产生实例

### 例题

* 下列的模板声明中，其中几个是正确的

    ```cpp
    1)template // 错误
    2)template<T1,T2> // 错误，缺少模板参数关键字 typename或class
    3)template<class T1,T2> // 错误，缺少模板参数关键字 typename或class
    4)template<class T1,class T2> // 正确
    5)template<typename T1,T2> // 错误，缺少模板参数关键字 typename或class
    6)template<typename T1,typename T2> // 正确
    7)template<class T1,typename T2> // 正确
    8)<typename T1,class T2> // 错误，没有template关键字
    9)template<typeaname T1, typename T2, size_t N> // 正确
    10)template<typeaname T, size_t N=100, class _A=alloc<T>> // 正确
    11)template<size_t N> // 正确
    ```

* 以下程序的输出结果为

    ```cpp
    template<typename Type>
    Type Max(const Type &a, const Type &b) {
        cout<<"This is Max<Type>"<<endl;
        return a > b ? a : b;
    }
    template<>
    int Max<int>(const int &a, const int &b) {
        cout<<"This is Max<int>"<<endl;
        return a > b ? a : b;
    }
    template<>
    char Max<char>(const char &a, const char &b) {
        cout<<"This is Max<char>"<<endl;
        return a > b ? a : b;
    }
    int Max(const int &a, const int &b) {
        cout<<"This is Max"<<endl;
        return a > b ? a : b;
    }
    int main() {
        Max(10,20); // "This is Max" // 可以直接匹配非模板参数，虽然形参是引用和权限缩小，但非模板参数仍然是最匹配的
        Max(12.34,23.45); // "This is Max<Type>" // 没有匹配的特化模板或显式实例化的实例，因此要调用模板
        Max('A','B'); // "This is Max<char>" // 已经给出了特化模板，直接匹配
        Max<int>(20,30); // "This is Max<int>" 直接进行了显式实例化，但因为给出了int条件下的特化，所以使用特化模板
        return 0;
    }
    ```
    
* 以下程序运行结果为

    ```cpp
    template<class T1, class T2>
    class Data {
    public: 
        Data() { cout << "Data<T1, T2>" << endl; }
    private:
        T1 _d1;
        T2 _d2;
    };
    template <class T1>
    class Data<T1, int> {
    public:
        Data() { cout << "Data<T1, int>" << endl; }
    private:
        T1 _d1;
        int _d2;
    };
    template <typename T1, typename T2>
    class Data <T1*,T2*> {
    public:
        Data() { cout << "Data<T1*, T2*>" << endl; }
    private:
        T1 _d1;
        T2 _d2;
    };
    template <typename T1, typename T2>
    class Data <T1&, T2&> {
    public:
        Data(const T1& d1, const T2& d2)
        : _d1(d1)
        , _d2(d2)
        {cout << "Data<T1&, T2&>" << endl;}
    private:
        const T1 & _d1;
        const T2 & _d2;
    };
    int main() {
        Data<double, int> d1; // "Data<T1, int>" 第二个参数int已经最匹配了 class Data<T1, int>
        Data<int, double> d2; // "Data<T1, T2>" 没有最匹配的特化模板，需要进行模板实例化
        Data<int*, int*> d3; // "Data<T1*, T2*>"
        Data<int&, int&> d4(1, 2); // "Data<T1&, T2&>"
        return 0;
    }
    ```

## *模板的分离编译*

模板不能进行分离编译，否则会因为两次编译导致链接错误

## *模板的优缺点*

* 优点
  * 模板复用了代码，节省资源，更快的迭代开发，STL库的诞生很大程度上得益于此
  * 增强了代码的灵活性
* 缺点
  * 模板只是将重复的代码交给编译器实现，因此模板也会导致代码膨胀问题，从而导致编译时间变长
  * 出现模板编译错误时，错误信息非常凌乱，不易定位错误

## *STL*

Standard Template Library 标准模板库是C++标准库 `std` 的重要组成部分，其不仅是一个可复用的组件库，而且是一个包含数据结构的软件框架

### STL版本

原始HP版本 -> PJ版本 -> RW版本 -> SGI版本（主要被使用的版本）

### STL六大组件

* Container 容器：容器就是各种常用的数据结构用C++实现，C++可以提供但C语言不能提供的原因主要是得益于C++提供了模板这种泛型编程方式
  * 序列式容器 Sequential container：底层为线性序列的数据结构，里面存储的是元素本身
    * string
    * vector
    * list
    * deque
  * 关联式容器 Associative container：存储的是 `<key, value>` 结构的键值对
    * map
    * set
    * multimap
    * multiset
  * c++11新增了array静态数组容器，和普通数组的主要区别在于对越界的检查更加严格，因为 `array[]` 的本质是函数调用 -- 运算符重载
* Iterator 迭代器：迭代器再不暴露底层实现细节的情况下，提供了统一的方式（即从上层角度看，行为和指针一样）去访问容器。屏蔽底层实现细节，体现了封装的价值和力量。迭代器被认为是algorithm和container的粘合剂，因为algorithm要通过迭代器来影响container

    <img src="迭代器初探.png" width="60%">

  * iterator
  * const_iterator
  * reverse_iterator
  * const_reverse_iterator
* Functor 仿函数
  * greator
  * less ...
* Algorithm 算法
  * find
  * swap
  * reverse
  * sort
  * merge ...
* Allocator 空间配置器
* Adapter 适配器/配接器：对容器进行复用
  * stack
  * queue
  * priority_queue

### STL缺陷

* 更新维护满
* 不支持线性安全你，在并发环境下需要自己加锁，且锁的粒度比较大
* 极度追求效率，导致内部比较复杂，如类型萃取，迭代器萃取
* STL的使用会有代码膨胀的问题

### 常用\<algorithm>库函数笔记

封装了容器通用的函数模板

* find
    > Returns an iterator to the first element in the range [first,last) that compares equal to val. If no such element is found, the function returns last.
* reverse：注意reverse范围左闭右开不包括last
    > Reverses the order of the elements in the range [first,last).
* sort
  * sort要使用随机迭代器，也就是vector那种支持随机访问的迭代器
  * sort的底层是不稳定快排，若要保证稳定性需要用 `stable_sort`

# string

string是一个特殊的顺序表，与顺序表的不同之处是它的操作对象是字符，并且最后一个数据默认是\0

## *标准库中的string常用接口*

### string类介绍

> `std::string`. Strings are objects that represent sequences of characters. String是一个char类型的动态增长数组

### 常见构造，string的构造函数是添加了 `\0` 的，满足C语言字符串规范

<img src="StringConstructor.png" >

### 容量操作

* `size` 与 `length` 是一样的，只是因为string类产生的比较早，用的是 `length`，后来出现了其他容器后统一规范为 `size`
* `reserve(size_t n)`：预留空间
* `clear`：清空有效字符
* `resize(size_t n, char c)`：将有效字符的个数改成n个，多出的空间用字符c填充（开空间+初始化）

### 访问及遍历操作

* `operator[]` 重载的意义：可以像操作数组一样用 `[]` 去读写访问对象
* `at` 和 `operator[]` 的区别是越界以后抛异常

### 迭代器 iterator：范围for的底层实现，范围for的底层就是直接替换迭代器

* iterator是属于容器的一个类，它的用法很像指针，可能是指针也可能不是
* string和vector不太使用迭代器，因为 `operator[]` 更方便，但是对于其他的容器如list、map、set来说只能用迭代器来访问。迭代器是所有容器的通用访问方式，用法类似
* C++中的迭代区间都是左闭右开 `[ )`，右开是因为方便遍历到 `\0` 时正好结束
* `begin` & `end`: return iterator to beginning/end
* `rbegin` & `rend`: return reverse iterator
* 四种迭代器
  * iterator/reverse_iterator
  * const_iterator/const_reverse_iterator

```cpp
// 迭代器示例
void PrintString(const string& str) {
    string::const_iterator it = str.begin();
//auto it = str.begin(); // auto自动推导
    while (it != str.end())
//*it = 'x';
        cout << *it++ << " ";
    cout << endl;

    string::const_reverse_iterator rit = str.rbegin();
    while (rit != str.rend())
        cout << *rit++ << " ";
    cout << endl;
}
```

### 修改操作

* `operator+=` 在字符串后追加字符串str
* `push_back` 和 `append`
  * 不如 +=，基本都是用 +=
  * 也可以使用迭代器，来控制插入字符的某一段

    ```cpp
    string s("hello");
    string str(" world");
    s.append(str.begin(), str.end());
    ```

* `insert` 效率很低
* `find` 和 `rfind`：返回 size_t，返回第一个/最后一个找到的位置，找不到则返回 `npos`，这时会解释为一个非常大的数，这是因为默认不存在这么大的字符串
* `substr` 从pos位置开始取n个字节长度的子串
* `c_str`：用来兼容C字符串，C不支持string类型

    ```cpp
    string filename("test.cpp");
    FILE* fout = fopen(filename.c_str(), "r");
    assert(fout);
    // ...
    ```

### 非成员函数

* getline：cin和scanf一样，输入以空格或换行符为间隔标志，getline和C语言中的fgetf函数功能一样，帮助取到一句有空格的字符串
* operator+：尽量少用，传值返回导致深拷贝效率低

### string 和其他数据类型之间的转换

* atoi
* stoi, stol, stof, stod, ...
* to_sring, to_wstring

## *string模拟实现中构造函数的问题*

### 错误写法一

```cpp
class String {
public:
    String(const char* str)
        :_str(str) // 给的是const char* 常量字符串，会报错
    {}
private:
    char* _str; // 因为_str是要提供接口被修改的，所以不能是const
    size_t _size;
    size_t _capacity;
}
```

直接给常量字符串会出很多问题，因此考虑开辟空间后复制

### 错误写法二：给nullptr

```cpp
String() // 提供一个空的
    :_str(nullptr)
    , _size(0)
    , _capacity(0) // _capacity不包括\0
{}
const char* c_str() const {
    return _str;
}

// 调用c_str接口
String s1;
cout << s1.c_str() << endl;
```

若空串给的是nullptr，调用的时候会发生对空指针解引用问题

### 正确写法一

```cpp
String() // 提供一个空的
    :_str(new char[1]) // 多new一个\0
    , _size(0)
    , _capacity(0) // _capacity不包括\0
{
    _str[0] = '\0';
}
String(const char* str)
    :_str(new char[strlen(str) + 1]) // 多new一个\0
    , _size(strlen(str))
    , _capacity(strlen(str)) // _capacity不包括\0
{
    strcpy(_str, str);
}
```

### 正确写法二：全缺省参数

```cpp
// 全缺省参数同样不能给nullptr，strlen直接对nullptr解引用了
String(const char* str="")
// String(const char* str="\0") // 其实这样给编译器还会再给一个\0，也就是\0\0
    :_str(new char[strlen(str) + 1]) // 多new一个\0
    , _size(strlen(str))
    , _capacity(strlen(str)) // _capacity不包括\0
{
    strcpy(_str, str);
}
```

strlen()是一个O(N)的函数，可以进一步改造

### 错误写法三

```cpp
class String {
    String(const char* str="")
        :_size(strlen(str))
        , _capacity(_size) // _capacity不包括\0
        , _str(new char[_capacity + 1]) // 多new一个\0
    {
        strcpy(_str, str);
    }
private:
    char* _str; 
    size_t _size;
    size_t _capacity;
}
```

这么写是错误的，因为初始化顺序是按照声明的顺序来，若这么写初始化列表_size，_capacity都是随机值。若更改声明的顺序会产生很多的维护问题

### 正确写法三

因为类成员都是内置类型，所以不使用初始化列表以避免初始化顺序依赖问题

```cpp
class string {
public:
    string(const char* str="") {
        _size = strlen(str);
        _capacity = _size;
        _str = new char[_capacity + 1];
        strcpy(_str, str);
    }
    ~string() {
        delte[] _str;
        _str = nullptr;
        _size = _capacity = 0;
    }
private:
    char* _str; 
    size_t _size;
    size_t _capacity;
}
```

## *string类的部分接口*

### `size` 和 `operator[]`

```cpp
size_t size() const {// 普通对象和const对象都可以使用
    return _size;
}

char& operator[](size_t pos) {
    assert(pos < _size); // string的遍历可以直接用 while(s[i++])，因为operator[]使用了_size，自动判断结束
    return _str[pos];
}
```

### stirng类中的迭代器就是原生指针

```cpp
typedef char* iterator; // 在string中的迭代器就是对原生指针的封装
typedef const char* const_iterator;
// 左闭右开
iterator begin() {
    return _str;
}
iterator end() {
    return _str + _size;
}
const_iterator begin() const {
    return _str;
}
const_iterator end() const {
    return _str + _size;
}
```

### string类的增删查改

* `reserve` 作用：避免最初的扩容开销、用来复用

    ```cpp
    char* my_strcpy(char* dest, const char* src);
    void reserve(size_t n) {
        if (n > _capacity) {
            char* tmp = new char[n + 1];
            strcpy(tmp, _str);
            delete[] _str;
    
            _str = tmp;
            _capacity = n;
        }
    }
    ```
    
* `push_back`：尾插一个字符
  * 直接实现

    ```cpp
    void push_back(char ch) {
        // 满了就扩容
        if (_size == _capacity)
            reserve(_capacity == 0 ? 4 : 2 * _capacity);
        _str[_size] = ch;
        _str[++_size] = '\0';
    }
    ```
    
  * `insert` 复用：`insert(_size, ch);`
  
* `append`：尾插一个字符串
  
  * 直接实现
  
    ```cpp
    void append(const char* str) {
        size_t len = strlen(str);
        // 满了就扩容
        if (_size + len > _capacity)
            reserve(_size + len);
        strcpy(_str + _size, str);
        // strcat(_str, str); // 不要用strcat，因为要找尾，效率比strcpy低
        _size += len;
    }
    ```
    
  * `insert` 复用：`insert(_size, str);`

* `operator+=`
  * `push_back` 复用

    ```cpp
    string& operator+=(char ch) {
        push_back(ch);
        return *this;
    }
    ```
    
  * `append` 复用
  
    ```cpp
    string& operator+=(const char* str) {
        append(str);
        return *this;
    }
    ```
  
* `insert`
  * 插入单字符

    ```cpp
    string& insert(size_t pos, char ch) {
        assert(pos <= _size); // =的时候就是尾插
        // 满了就扩容
        if (_size == _capacity)
            reserve(_capacity == 0 ? 4 : 2 * _capacity);
        // 挪数据
        size_t end = _size + 1;
        while (end > pos) {
            _str[end] = _str[end - 1];
            --end;
        }
        // 插入字符
        _str[pos] = ch;
        ++_size;
    
        return *this;
    }
    ```
  
    <span id="经典隐式类型转换错误">在这里会有两个坑</span>

    * 一开始的思路是如下的，将end定义为size_t，但此时当pos=0时，因为end是一个无符号数，0-1会直接让end变成一个很大的数字，因此程序会陷入死循环。
    * 然后考虑把end改为定义成int，但是当执行 `end>=pos` 这个运算时，由于==隐式类型转换==的存在，又会把end提升为一个无符号数，又出现了死循环。因此最后的设计是上面的
  
    ```cpp
    // 不好的设计1
    // ...
    size_t end = _size;
    while (end >= pos) { //隐式类型转换
        _str[end+1] = _ str[end]
        --end;
    }
    ```

    ```cpp
    // 不好的设计2
    // ...
    int end = _size;
    while (end >= (int)pos) {
        _str[end+1] = _ str[end]
        --end;
    }
    ```
  
    这样设计不好，应该将位置下标设计为size_t，挪动设置为 `_str[end] = _str[end-1]` 防止越界
  
  * 插入字符串
  
    ```cpp
    string& insert(size_t pos, const char* str) {
        assert(pos <= _size);
        size_t len = strlen(str);
        // 满了就扩容
        if (_size + len > _capacity)
            reserve(_size + len);
        // 挪数据
        size_t end = _size + len;
        while (end >= pos + len) {
            _str[end] = _str[end - len];
            --end;
        }
        // 插入字符串
        strncpy(_str + pos, str, len);
        _size += len;
        return *this;
    }
    ```

## *深拷贝 Deep Copy*

### 浅拷贝 Shallow Copy

浅拷贝问题在之前的默认拷贝构造函数和默认赋值运算符重载中已经遇到过，默认拷贝构造是浅拷贝，对于str这种开辟了空间的变量，指针指向的是同一个内存空间。当析构的时候会释放同一个内存空间多次。为解决这一个问题需要用户需要手动写深拷贝，即创造两份内存空间

### 深拷贝传统写法

```cpp
string(const string& s)
    :_str(new char[s._capacity + 1])
    , _size(s._size)
    , _capacity(s._capacity)
{
    strcpy(_str, s._str);
}
```

对于operator=，不考虑dest空间是否足够容纳src，直接将dest释放掉，然后给一个新new的src容量的空间tmp，然后进行拷贝

```cpp
string& operator=(const string& s) {
    if (this != &s) {// 防止自己给自己赋值
        char* tmp = new char[s._capacity + 1]; // 先new tmp再 delete[] _str 的原因是因为防止new失败反而破坏了_str
        strcpy(tmp, s._str);
        delete[] _str;
        _str = tmp;
        _size = s._size;
        _capacity = s._capacity;
    }
    return *this;
}
```

### 现代写法：安排一个打工人

```cpp
void swap(string& tmp) {
    std::swap(_str, tmp._str); // sdt::表明调用的是std库中的swap，否则编译器会优先调用局部域中的swap（也就是自己变成迭代了）
    sdt::swap(_size, tmp._size);
    std::swap(_capacity, tmp._capacity);
}

string(const string& s)
    :_str(nullptr)
    , _size(0)
    , _capacity(0) // 不给初始值的话，delete随机值会崩溃
{
    string tmp(s._str);
    swap(tmp); // this->swap(tmp); 用的是自己写的当前域下的swap，优先访问局部域
}
```

* `operator=`的两种写法
  * 利用tmp

    ```CPP
    string& operator=(const string& s) {
        if (this != &s) {// 防止自己给自己赋值
            string tmp(s);
            ::swap(*this, tmp);
        }
        return *this;
    }
    ```
    
  * 传值传参，直接让s顶替tmp当打工人

    ```cpp
    string& operator=(string s) {
        swap(s);
        return *this;
    }
    ```
  
* <img src="swapRef.png" width="60%">

    std提供的swap函数代价很高，需要进行3次深拷贝，因为需要借助中间变量。所以可以自己写swap，通过复用库里的`std::swap`来交换一些开销小的内容

* 下面这种写法是错误的，会报栈溢出，因为swap和赋值会重复调用，造成迭代栈溢出

    <img src="swap栈溢出.png" width="60%">

## *写时拷贝 Copy-on Write COW*

### 浅拷贝的问题及其解决方案

* 析构两次。解决方案：增加一个引用计数，每个对象析构时，--引用计数，最后一个析构的对象释放空间
* 一个对象修改影响另外一个对象。解决方案：写时拷贝，本质是一种延迟拷贝，即谁去写谁做深拷贝，若没人写就可以节省空间了

### 引用计数+写时拷贝

# vector和list

## *vector 顺序表*

### 定义

<img src="vectorDef.png">

* vector是一个类模板，对应的是数据结构中的顺序表/数组。比如声明一个存储int数据的数组 `vector<int> v1;`

* vector成员

  <img src="vectorMember.png" width="50%">

### vector迭代器在 insert 和 erase 中的失效问题

* 问题一：当insert（在pos前插入数据）中进行扩容时会出现迭代器失效问题
  * 旧的pos发生了越界造成了野指针问题。任意下图所示，在扩容后pos指针不会有变化，已经不处于新开的空间中了，即不处于_start和_finish之间
  
    <img src="迭代器失效问题1.png">
  
  * 修正：计算pos和_start的相对位置，扩容后令原pos重新指向
  
* 问题二：在p位置修改插入数据以后不要访问p，因为p可能失效。这是因为调用insert的时候pos是传值传参，内部对pos的修改不会影响实参。STL库中的实现也没有给pos设置为传引用传参，因为这又会引起一些其他的问题；erase有可能会出现缩容的情况，但是很少，此时也不要在erase后解引用访问

* 问题三：因为数据挪动，pos在insert/erase之后位置发生了改变。
  * 问题代码

    ```cpp
    void erase(iterator pos) {
        assert(pos >= _start && pos < _finish);
        // 从前往后挪动数据
        iterator begin = pos + 1;
        while (begin < _finish) {
            *(begin - 1) = *begin;
            begin++;
        }
        _finish--;
    }
    void test() {
        while (it != v.end()) {
            if (*it % 2 == 0)
                v.erase(it);
            it++;
        }
    }
    ```
    
    <img src="迭代器失效问题3.png" width="60%">
    
  * 修正：每次insert/erase之后要更新迭代器，STL规定了erase/insert要返回删除/插入位置的下一个位置迭代器

    ```cpp
    // STL规定了erase要返回删除位置的下一个位置迭代器
    iterator erase(iterator pos) {
        assert(pos >= _start && pos < _finish);
        // 从前往后挪动数据
        iterator begin = pos + 1;
        while (begin < _finish) {
            *(begin - 1) = *begin;
            begin++;
        }
        _finish--;
        return pos; // 返回删除数据的下一个位置 还是pos
    }
    void test() {
        while (it != v.end()) {
            if (*it % 2 == 0)
                //v.erase(it);
                it = v.erase(it);
            else
                it++; // erase之后更新迭代器
        }
    }
    ```
  
* 总结：insert/erase之后不要直接访问pos。一定要更新，直接访问可能会出现各种意料之外的结果，且各个平台和编译器都有可能不同。这就是所谓的迭代器失效

### 拷贝构造与高维数组的浅拷贝问题

* 高维数组或容器里存的数据是另外的容器的时候，就会涉及到更深层次的深拷贝问题

    <img src="memcpy浅拷贝问题.png">

    原因是因为对于类似于在string的深拷贝实现使用的是memcpy函数，若容器中村的还是容器类的数据类型，那么它进行的仍然是浅拷贝。上图是利用杨辉三角OJ题做的试验，存储的数据类型是 `vector<vector<int>>`，可以看到左图虽然外层vector实现了深拷贝，但内容的数据仍然是浅拷贝。右边进行了修正

* 深拷贝构造的写法
  * 传统写法

    ```cpp
    // 传统写法1 
    vector(const vector<T>& v) {// v2(v1)
        _start = new T[v.size()]; // 开v.capacity()个空间也可以，各有各的优势和劣势
        // 不能使用memcpy，memcpy也是浅拷贝，当出现类似与vector<vector<int>> 这种多维数组就会有问题
        // memcpy(_start, v._start, sizeof(T) * v.size());
        for (size_t i = 0; i < v.size(); i++)
                _start[i] = v._start[i];
        _finish = _start + v.size();
        _end_of_storage = _start + v.size();
    }
    // 传统写法2
    vector(const vector<T>& v)
        :_start(nullptr)
        , _finish(nullptr)
        , _end_of_storage(nullptr)
    {
        reserve(v.size());
        for (const auto& e : v) // 若是vector则传值要深拷贝，因此用引用
            push_back(e); // push_back 会自动处理_finish和_end_of_storage
    }
    ```
    
  * 现代写法
  
    ```cpp
    // 现代写法
    // 提供一个迭代器区间构造
    template <class InputIterator>
    vector(InputIterator first, InputIterator last)
        :_start(nullptr)
        , _finish(nullptr)
        , _end_of_storage(nullptr) {
        while (first != last) {
            push_back(*first);
            first++;
        }
    }
    void swap(vector<T>& v) {
        std::swap(_start, v._start);
        std::swap(_finish, v._finish);
        std::swap(_end_of_storage, v._end_of_storage);
    }
    vector<T> operator=(vector<T> v) {
        swap(v);
        return *this;
    }
    vector(const vector<T>& v)
        :_start(nullptr)
        , _finish(nullptr)
        , _end_of_storage(nullptr)
    {
        vector<T> tmp(v.begin(), v.end());
        swap(tmp);
    }
    ```

## *list 链表*

### list的特殊operation

* `splice`
* `remove`
* `srot` 排升序：属于list的sort和algorithm库中的sort的区别在于list的空间不连续，而algorithm的适用对象是连续空间，不能用于list。这是因为algorithm的底层qsort需要实现三数取中法。list的sort底层是MergeSort
* list排序 VS vector排序：大量数据的排序vector效率远高于list

### list的erase迭代器失效

* list的插入是在pos位置之前插入，它采取的方式新建一个newcode，然后改变指针指向，并没有挪动数据，因此迭代器的位置不会改变也不会出现野指针问题
* 但是当erase时迭代器仍然会失效：pos被erase，也就是被free了，这时候访问就是野指针问题。因此erase的返回值也是iterator，方便进行更新

### list的数据结构与迭代器模拟实现

* C++中更倾向于使用独立的类封装，而不是使用内部类。因此list的设计采用了 `list_node`，迭代器和 `list` 总体分别封装成独立的类（这里 `list_npde` 和迭代器 直接用了 `struct`，因为要将类成员设置为公有，供 `list` 使用）
* C++用**带头双向循环链表**来实现list

    ```cpp
    template<class T>
    struct list_node {// 用struct和STL库保持一致
    // 同时struct默认公有，可以让其他成员调用
        list_node(const T& x = T()) // T() 为数据类型的默认构造函数
            :_data(x)
            , _next(nullptr)
            , _prev(nullptr)
        {}
        T _data;
        list_node<T> *_next;
        list_node<T> *_prev;
    }
    ```
    
* 迭代器类封装

    ```cpp
    template<class T, class Ref, class Ptr> // 准备多个模板参数是为了const_iterator复用
    struct _list_iterator {
        typedef list_node<T> Node;
        typedef _list_iterator<T, Ref, Ptr> iterator;
    
        Node *_node; // 迭代器类的唯一成员变量，实际上迭代器就是封装的Node*
    
        _list_iterator(Node* _node)
            :_node(node)
            {}
        // 对迭代器结构体的运算符重载：!= == * ++ -- ->
        bool operator!=(const iterator& it) const;
        bool operator==(const iterator& it) const;
        Ref operator*();
        Ptr operator->();
        iterator& operator++() // 前置++
    }
    ```
    
  * 不需要析构函数，默认的析构函数不会处理Node* 指针。这是合理的，因为不可能在用户使用迭代器操作后，把list中的节点给销毁了
  * 也不需要写拷贝构造，因为拷贝得到的迭代器必然要指向同一个地址，因此默认的浅拷贝就够了。而浅拷贝不会报错的原因是因为轮不到迭代器进行析构，迭代器只是封装，list会将所有的一块析构掉
  * `_list_iterator` 不支持 `+ += - < >` 等操作符，因为空间地址不连续，这些操作没有意义
  * const_iterator 和 iterator 的区别是是否能够修改数据，即是返回T& 还是 const T&。不能使用函数重载，因为仅仅只有返回值不同的话是不构成函数重载的。因此考虑使用类模板复用的方式
  * `++` 等操作的返回值为iterator的原因是因为也要支持 const_iterator 等其他迭代器，即要支持 `typedef _list_iterator<T, Ref, Ptr> iterator` 模板实例化后得到的所有迭代器
  
* list总体

    ```cpp
    template<class T>
    class list {
        typedef list_node<T> Node; // Node是只给当前类使用的封装，因此设置为私有
    public:
        typedef _list_iterator<T, T&, T*> iterator; // 普通迭代器
        typedef _list_iterator<T, const T&, const T*> cosnt_iterator; // const迭代器 
        typedef __reverse_iterator<iterator, T&, T*> reverse_iterator; // 反向迭代器
        typedef __reverse_iterator<const_iterator, const T&, const T*> const_reverse_iterator // const反向迭代器
        // 迭代器+增删查改接口
    private:
        Node* _head; // 类成员变量只有一个哨兵位头结点
    }
    ```
    
* 迭代器中特殊的运算符重载 `->`
  * 考虑当list中存的是一个自定义类型Pos

    ```cpp
    struct Pos {
        int _a1;
        int _a2;
        Pos(int a1 = 0, int a2 = 0)
            :_a1(a1)
            , _a2(a2)
        {}
    };
    
    list<Pos> lt;
    lt.push_back(Pos(10, 20));
    lt.push_back(Pos(10, 21));
    ```
  
  * `T* operator->()` 返回的是lt中存储的一个结构体指针*Pos，若要取到其实中的数据应该要 `it->->_a1`，但编译器为了提高可读性，进行了特殊处理，即省略了一个 `->`，自动取到的就是Pos中的一个数据。因此当lt中存储的是自定义类型或者内置类型时，`->` 都可以看作是迭代器指针取数据
  
    ```cpp
    T& operator*()
        return _node->_data;
    T* operator->()
        return &(operator*()); 
    ```

### list的反向迭代器，采用适配器（复用）的方向进行设计

* 迭代器按功能分类
  * forward_iterator：只支持++，不支持--：比如forward_list、unordered_map、unordered_set
  * bidirectional_iterator：既支持++，也支持--：比如list、map、set
  * random_access_iterator：不仅支持++--，还支持+-，比如vector、deque
* 实现方法
  * 普通思维：拷贝一份正向迭代器，对其进行修改
  * STL的设计：对iterator进行复用。反向迭代器里封装的是正向迭代器，正向迭代器里封装的是指针或节点

    ```cpp
    template<class Iterator, class Ref, class Ptr>
    struct __reverse_iterator {
        Iterator _curr; // 类成员，当前的正向迭代器
        typedef __reverse_iterator<Iterator, Ref, Ptr> RIterator;
        __reverse_iterator(Iterator it)
            :_curr(it)
        {}
    }
    
    RIterator operator++() {// 前置++，++方向置为反向
        --_curr;
        return *this;
    }
    RIterator operator--();
    Ref operator*(); // operator*的实现比较特殊，采用了和iterator的对称设计，见下方
    Ptr operator->();
    bool operator!=(const RIterator& it);
    ```
  
* `Ref operator*()` 的特殊设计：将 `end` 与 `rbegin` 以及 `begin` 与 `rend` 设计为对称关系

    <img src="end与rbegin对称关系的底层设计.png">

* 只要实现了正向迭代器，那么 `reverse_iterator` 可以复用到其他的容器上，除了 `forward_list`，`unordered_map` 和 `unordered_set` 不能被复用，因为这些容器的迭代器不支持 `--`

# stack & queue 栈和队列

## *deque容器*

<div align="center"><img src="deque.png" width="40%"></div>

* deque每次既不是开一个节点，也不是进行realloc，而是开多个可以存多个数据的小buffer
* Double ended queue 双端队列融合了vector和list的优点，既支持list的任意位置插入删除，又支持vector的下标随机访问
* 设计缺陷：
  * 相比vector，`operator[]` 的计算稍显复杂，大量使用会导致性能下降
  * 中间插入删除效率不高
  * 从底层角度看迭代器的设计非常复杂

    <img src="dequeIterator.png" width="75%">

    * curr为当前数据
    * first和last表示当前buffer的开始和结束
    * node反向指向中控位置，方便遍历时找下一个buffer

* 结论：相比vector和list，deque非常适合做头尾的插入删除，很适合去做stack和queue的默认适配容器；但如果是中间插入删除多用list，而随机访问多用vector

## *仿函数 Functor*

在优先级队列中需要比较元素的大小以进行建堆。但在建堆的逻辑中，比较大小是写死的，若此时要将大堆切换成小堆，就需要进入源码中改比较的顺序。这在实际中是不可能的。因此在类实例化时候就需要提供一个“开关”进行控制

仿函数是一个类/结构体，它的目标是使一个类的使用看上去像一个函数。 其实现就是类中实现一个 `bool operator()`（**`()`也是运算符！**)，这个类就有了类似函数的行为，就是一个仿函数类了

模板参数只允许是类，因此仿函数就可以很好的放入模板中进行泛型编程

```cpp
namespace wjf { //9.24的课
    template<class T> //less和greater在std中都被提供了
    struct less {//用class也行，但要设置为public
        bool operator()(const T& l, const T& r) const {
            return l < r;
        }
    };
    template<class T>
    class greater { //用class也行，但要设置为public
        public:
        bool operator()(const T& l, const T& r) const {
            return l > r;
        }
    };
}

//调用，二者等价
wjf::less<int> lsFunc; //仿函数是一个类，别忘了实例化
cout << lsFunc(1, 2) << endl; //看起来就像是一个普通的函数调用
cout << lsFunc.operator()(1, 2) << endl; //本质就是调用运算符重载
```

要注意的点是类实例化时要传入的是类型，而函数重载时传入的则是对象。比如`std::sort()`是一个函数重载，它在调用的时候要传入一个 `Compare comp`的对象（`std::greater<int>()` 对象有括号），而对于priority_queue类实例化则传入的是 `less<int>`这种类型

```cpp
vector<int> v;
sort(v.begin(), v.end(), less<int>()); // 传入一个匿名对象

priority_queue<int, vector<int>, Compare = less<int>> pq; // 传入一个类型
// 优先级队列的STL参数设计有写问题，因为一般不太会更改默认容器vector，但是有可能更改排序方式，即Compare，因此应该将Compare放在第二位比较好
```



## *stack & queue适配器*

### stack

<img src="stackDef.png">

* 适配器 adapter 是一种设计模式，该种模式是将一个类的接口转换成客户希望的另外一个接口。其实就是复用以前的代码
* stack默认复用的容器是deque，也可以复用vector和list

### queue

<img src="queueDef.png">

queue 和 stack 一样都是默认复用 deque 的适配器

### priority_queue 优先级队列容器适配器

<img src="priorityQueueDef.png">

* `priority_queue` 可以实现最大值/最小值在队头。仿函数Compare默认为`std::less`，大数优先，T要支持 `operator<()`；Compare为`std::greater`时为小数优先，T要支持 `operator>()`
* `priority_queue` 的底层是堆，但用vector来控制下标
* `priority_queue` 的默认容器为vector，这是因为要进行大量的下标随机访问，用deque或list都不好

# 继承 Inheritance

## *继承的概念及定义*

继承是类层次设计的复用

### 继承格式

```cpp
class Person {
public:
    void Print();
protected:
    string _name;
    int _age;
};

class Student : public Person {
protected:
    int _stuID;
};
```

其中Person被称为父类/基类 Base class，而Student被称为子类/派生类 Derived class，Person前的访问限定符规定了派生类的继承方式

### 继承关系和访问限定符

类成员/继承方式| public继承 | protected继承 | private继承
:-:|:-:|:-:|:-:
基类的public成员|派生类的public成员|派生类的protected成员|派生类的private成员
基类的protected成员|派生类的protected成员|派生类的protected成员|派生类的private成员
基类的private成员|在派生类中不可见|在派生类中不可见|在派生类中不可见

* 不可见/隐藏 hide 是指基类的私有成员虽然还是被集成到了派生类对象中，但是语法上限制派生类对象不管在类内还是类外都**不能**去访问它，这种继承的使用情况极少
* 当省略继承方式时，class的默认继承方式是private，而struct的默认继承方式是public
* 实际运用中一般只会使用public继承，然后通过类成员是用public还是protected修饰来区分类内外是否可以访问

### 继承中的作用域

* 在继承体系中父类和子类都有独立的作用域
* 父类和子类中有同名成员时，子类成员将直接屏蔽对父类中同名成员的直接访问，这种情况叫做隐藏 hide，也叫做重定义。若在子类中想要访问父类中被隐藏的成员，可以通过指定父类域的方式，即 `父类::父类成员` 显式访问
* 对于成员函数只要函数名相同就构成隐藏。注意在不同类域中的同名函数不是函数重载
* 实际中最好不要定义相同命名的成员

## *派生类的默认成员函数*

### 基类和派生类对象赋值转换：切片 slicing

<img src="基类和派生类对象赋值转换.png">

* 每一个子类对象都是一个特殊的父类成员。实际上如上图所示，每一个子类对象中，都首先存放着父类成员
* 子类对象可以赋值给父类对象、父类的指针和父类的引用，这种赋值方式称为切片/切割
* 反过来父类对象不能赋值给子类对象
* 父类的指针或者引用可以通过强制类型转换赋值给子类的指针或者引用。但是必须是父类的指针是指向派生类对象时才是安全的。若父类是多态类型，可以使用RTTI的dynamic_cast来进行识别后进行安全转换

### 子类的默认构造函数、析构函数

* 自己的成员还是一样调用自己的默认构造函数
* 继承自父类的成员需要调用父类的构造函数进行初始化
* 若父类仅提供了不带缺省值的构造函数或者说没有可用的默认构造函数时，这个时候就需要子类为继承的父类成员提供显式构造。注意，必须调用整个父类的构造函数而不是为父类成员单独赋值，这被称为合成版本
* 析构函数同上。但析构函数有一个特殊点：子类的析构函数跟父类的析构函数构成隐藏，这是由于多态的需要，析构函数会被编译器统一处理成 `destructor()` 从而构成了隐藏，在调用父类的析构函数时需要指定类域
* 为了保证先子类析构，再父类析构的正确析构顺序，编译器会在子类析构后自动调用父类析构函数，用户不应该显式地给出父类的析构函数

    ```cpp
    class Student : public Person {
    public:
        Student(const char* name, int num)
            :Person(name)
            , _num(num)
        {}
    protected:
        int _num;
    };
    ~Student() {
        //Person::~Person(); // 指定类域，避免隐藏
        // ...
    }
    ```

### 子类的默认拷贝构造、`operator=`

* 自己的成员还是一样调用自己的默认拷贝构造函数 -- 内置类型浅拷贝，自定义类型调用它的拷贝构造
* 继承自父类的成员需要调用父类的拷贝构造函数进行初始化
* 当子类中需要深拷贝时，需要显式提供子类拷贝构造函数：利用赋值转换切片原理为父类赋值，或者利用强制类型转换取数据 `*(Person)*this`
* 复制运算符重载同上，只是要注意显式调用父类的 `operator=` 以避免隐藏

    ```cpp
    // ...
    Student(const Student& s) // 子类显式的拷贝构造
        :Person(s) // 赋值转换切片
    {}
    Student& operator=(const Student& s) {
        if (this != &s) {
            Person::operator=(s); // 显式调用父类的operator=
            _num = s._num;
        }
        return *this;
    }
    ```

## *友元和静态成员的继承*

* 友元关系不能被继承，也就是说父类的友元不能访问子类私有和保护成员
* 基类定义了static静态成员，则整个继承体系里面只有一个这样的成员，无论派生出多少个子类，都只有一个static成员实例
* 不能说父类或子类对象中包含了所有父类或子类的成员变量，静态变量就不属于父类或子类对象，而是属于整个类

## *菱形继承与菱形虚拟继承*

### 继承分类

* 单继承 Single Inheritance：一个子类只有一个直接父类时称这个继承关系为单继承
* 多继承 Multiple Inheritance：一个子类有两个或以上直接父类时称这个继承关系为多继承

    <img src="multipleInheritance.png">

* 菱形继承 Diamond Inheritance：是多继承的一种特殊情况

    <img src="diamondInheritance.png">

### 多继承的指针偏移问题

<img src="多继承的指针偏移问题.png">

* 对于同一个多继承的子类对象，当进行切片时，父类的继承顺序会有影响。如上例，由于先继承了Base1，后继承了Base2，所以根据栈的性质Base1对象放在低地址，而Base2产生了指针偏移，由于Base1中只有一个int对象，因此Base2放在离Base1有4个字节远的高地址
* 当分别进行切片时，p1看到的就是头四个字节，p2看到的范围为p1+4，而d则看到了全部，这就是切片的原理

### 菱形继承问题及虚拟继承

* 菱形继承的二义性问题：通过指定类作用域解决

    ```cpp
    Assistant at;

    //// 菱形继承的二义性问题
    //at._name = "张三";
    at.Student::_name = "张三";
    at.Teacher::_name = "李四";
    ```

* 数据冗余问题：同样的一个人有多个年龄、性别？Student和Teacher的类成员已经有了 `_age` 和 `_sex`，但经过继承和修改后，子类可能有不同的数据，这是多继承带来的问题，单继承就没有这个问题，因为单继承的父类属性只有一份。需要通过菱形虚拟继承来解决

    <img src="diamondInheritanceVirtual.png">

* 实际中都会避免定义成菱形继承，菱形继承的情况很少见，比如C++ IO流库中的iostream，底层机制会非常复杂
* 菱形虚拟继承的底层：取决于虚继类_a大不大，若很大就能节省空间，比如说当_a是一个4000字节的类对象时，我们只需要增加一个4字节虚继表指针就可以换来少存储一个冗余的_a对象
  * 若不使用虚拟继承，则编译器会根据继承顺序，算好空间后依次存放数据

    <img src="diamondInheritance底层.png" width="80%">

  * 使用了虚拟继承，此时在原来存放着继承量的地方就会改为存放指针（该指针空间称为虚继表），指针中存放的是一个偏移量，这个偏移量阐明了继承值和虚拟继承后的共同虚继类之间差了几个字节，从而帮助编译器找到

    <img src="diamondInheritanceVirtual底层.png">

  * 同时在虚拟继承后，设置为虚拟继承的类的底层存储方式也要发生变化。若不保持一致的模型，则会对其他指针的访问产生歧义，B类原来只需要开8个字节的大小，现在为了保存一个虚继表指针，变成了12字节

    <img src="虚继承对类底层的影响.png">

* 总结：当大量数据时，虚拟继承对底层的改造是有可能影响数据的，尽量不要设计出菱形继承和虚拟继承

## *继承和组合*

* 继承：is-a
* 组合：has-a

# 多态 Polymorphism

## *多态的定义及实现*

### 构成多态的两个条件

多态是在不同继承关系的类对象，去调用统一函数，却产生了不同的行为。构成多态有两个条件

* 必须通过父类的指针或者引用调用虚函数
* 被调用的函数必须是虚函数，且子类必须对父类的虚函数进行重写

### 虚函数重写的条件 Override of virtual function

* 虚函数是被 `virtual` 关键字修饰的类成员函数
* 对虚函数的重写/覆盖：子类中有一个跟父类完全相同的虚函数（即子类虚函数与父类虚函数的返回值类型、函数名字、参数列表完全相同，不要求参数的缺省值是否相同），称子类的虚函数重写了父类的虚函数。不符合重写就是隐藏关系

### 虚函数重写的三个例外

* 子类虚函数不加 `virtual`，依旧构成重写，这是因为编译器认为子类从父类那里已经继承了virtual，这种特例其实是为了析构函数的重写所准备的。实际中最好也加上构成同一的形式。

    ```cpp
    class Person {virtual void BuyTicket();};
    class Student {void BuyTicket();};
    ```

  * 析构函数的重写：若父类的析构函数为虚函数，此时无论子类析构函数是否定义或是否加virtual，二类都构成重写。这是因为编译器对析构函数的特殊处理，即编译后析构函数的名称同一处理为destructor

    ```cpp
    // 析构函数构成虚函数重写
    class Person {virtual ~Person() {}};
    class Student {~Student () {};};
    ```

  * 问题：为什么建议在继承中析构函数定义成虚函数？

    ```cpp
    Person *ptr1 = new Person;
    Person *ptr2 = new Student; // 在某些情况下使用了切片
    delete ptr1;
    delete ptr2; // 若此时析构不构成多态，则此时会调用Person的析构函数，而不是Student的析构函数
    ```

* 协变返回类型 Covariant returns type：返回值可以不同，但要求必须是父子关系的指针或者引用。注意：可以不是虚函数所属类的父子关系，只要是构成父子关系的类都可以

    ```cpp
    class Person {virtual Person* BuyTicket();};
    class Student {virtual Student* BuyTicket();};
    
    class Person {virtual A* BuyTicket();}; // B是A的子类，A和B与Person和Student无关
    class Student {virtual B* BuyTicket();};
    ```

### C++ 11 override 和 final

* <span id="final">若一个虚函数不想被重写，可以被定义为 final，但这种场景极少。因为虚函数的目标就是为了重写 `virtual void BaseFunc() final {};`</span>
* override 检查子类虚函数是否完成重写，若没有完成重写就报错 `virtual void DerivedFunc() override {};`

## *抽象类/接口类*

### 概念

* 纯虚函数 Pure virtual function：在虚函数的后面写上 `=0`
* 包含纯虚函数的类叫做抽象类/接口类。抽象类不能实例化出对象，子类继承后也不能实例化出对象，只有重写纯虚函数后才能实例化出对象。一个大的概念就很适合定义成抽象类，比如树、花、车等，然后再定义具体的子类，比如桃树、玫瑰花、BYD等
* 纯虚函数规范了子类必须重写，相当于规定了子类必须实现哪些接口，这也体现出了接口继承的概念

```cpp
class Car {
public:
    virtual void Drive() = 0;
}

class BYD : public Car {
public:
    virtual void Drive() {/*override*/}
}
```

### 实现继承与接口继承

* 实现继承 Implementation inheritance：普通函数的继承是一种实现继承，子类继承了父类的普通函数，继承的是函数的实现
* 接口继承 Interface inheritance：虚函数的继承是一种接口继承，子类继承的仅仅是父类虚函数的接口，因此要求返回值类型、函数名和参数列表完全相同。目的就是为了重写以达成多态。所以如果不需要实现多态，就不要把函数定义成虚函数

## *多态原理介绍：以单继承为例*

总结：满足多态条件时（函数重写+父类指针/引用调用），会产生虚表存储虚函数的指针，调用时取对象自己的虚表里找到重写的多态函数

### 虚函数表/虚表 virtual function table

* 虚函数在编译好后也是放在公共代码段中，虚表中放的只是虚函数指针
* 虚表的本质是一个虚函数指针数组，一般情况这个数组最后放了一个nullptr
* 虚表地址与重写的关系，这种关系的处理与编译器有关，下面是VS的情况
  * 若完成重写，同一个类的不同对象共用一张虚表
  * 若父子类都完成重写，则父子类的虚表不同，虚函数地址也不同
  * 若父类定义了虚函数，然而子类并没有重写，则父子类的虚表仍然不同，但其存储的虚函数指针是相同的
  * 若子类有父类没有的独立的虚函数，该虚函数的函数指针也会被放进虚表里，但VS中的监视窗口是看不到这个函数指针的；反过来若父类有子类没有就看得到。可以通过设计打印函数来观察

    <img src="虚表地址与重写的关系.png" width="80%">

* 虚表和菱形继承产生的虚继表是完全不同的东西，注意区别。虚表应对多态问题，虚继表应对菱形继承的数据冗余和二义性问题。当既是菱形继承，又发生了多态时，二者会同时产生，变得非常复杂

### 多态原理

<img src="多态原理.png">

* 上图中得到的一些虚表结论
  * 若满足虚函数重写的条件，父类和子类都会存放虚表，由上图绿框所标识，两个虚表的地址不同；若不满足虚函数重写，则不会生成虚表
  * Func1虚函数进行了重写，因此实际上成为了两个不同的函数
  * Func2函数虽然是虚函数，但子类并没有对其进行重写，因此正常地被子类所继承，因此虚表中Func2的地址是相同的
  * Func3函数没有被定义为虚函数，就没有被放入虚表中
  * 用汇编语言看也证实了，构成多态调用Func1时，实际上call的是eax寄存器，其中存放的是虚表的指针；而对于一般函数Func3则直接call函数的地址
  * 满足多态以后的函数并不是在编译时确定的，而是运行起来以后到对象的栈中找到的，所以指向谁调用谁，这也被称为运行时决议 Execution-time resolution；不满足多态的函数地址则在编译时就被确定了，这也被称为编译时决议 Compile-time resolution
* 多态的本质原理：若符合多态的两个条件，那么调用时，会到指向对象的虚表中找到对应的虚函数地址，进行调用。比如在上述代码中当传入d对象时，会调用d对象的栈
* 普通函数调用：编译连接时确定函数的地址，运行时直接调用
* 必须通过父类的指针或者引用调用虚函数原因在于利用了切片原理，由于子类对象继承了父类的结构并把父类结构放在最前面，因此传入子类进行切片后编译器看到的也是父类的结构，但里面的虚表保存的是子类中的虚函数位置，这就产生了传入什么对象就调用什么对象的虚函数。

### 动态绑定与静态绑定

## *多继承关系中的虚函数表*

### 打印虚表

```cpp
typedef void(*VFPTR)(); // 因为上面的成员函数都是空函数指针，所以对空函数指针重定义
// 即使虚函数的返回类型不是void，或者有其他参数（这种情况下VS2019也会报错），它的函数指针也会被强转为空函数指针
void PrintVFTable(VFPTR table[]) {
    for (size_t i = 0; table[i] != nullptr; i++) {// VS中虚表以nullptr结尾，而Linux没有这个设计
    //  for (size_t i = 0; i < 3; i++) // VS中虚表以nullptr结尾，而Linux没有这个设计
        printf("vft[%d]: %p ", i, table[i]);
        //table[i]();
        VFPTR pf = table[i];
        pf();
    }
}
int main() { PrintVFTable((VFPTR*)(*(int*)(&s1))); /*...*/}
```

### 多继承中的虚函数表

* 多继承中子类中独立的虚函数会按照继承顺序放到第一个继承的父类的虚表中
* Derive的func1对两个父类都进行重写，理论上应该共用一份func1，但两个虚表中的func1地址却不一样。这是因为VS中jump指令对func1进行了多次封装，最后实际上调用的都是同一个func1
* 菱形继承、菱形虚拟继承中的虚函数表：尽量不要写出这样的继承

## *继承和多态的一些面试问题*

* 如何定义一个不能被继承的类
  * 方法1：C++98：1、父类构造函数私有 2、子类对象实例化，无法调用构造函数

      ```cpp
      class A {
      private:
          A()
          {}
      protected:
          int _a;
      };
      
      class B : public A
      {};
      
      int main() {
          B b;
          return 0;
      }
      ```
      
  * 方法2：C++11新增了一个final关键字（最终类） `class A final`，此时 class A 不能用作基类
* 什么是多态？
* 什么是重载、重写/覆盖、重定义/隐藏？
* 多态的实现原理：满足多态条件时（函数重写+父类指针/引用调用），会产生虚表存储虚函数的指针，调用时取对象自己的虚表里找到重写的多态函数
* inline函数可以是虚函数吗？内联函数没有地址不能被放进虚函数，这和虚函数是互斥的。但是是可以实现的，若设置为virtual，编译器会忽略inline设置，因为inline是一个对编译器的建议性关键字，优先级很低
* 静态成员可以是虚函数吗？不可以，因为静态函数都是编译时决议，无法访问虚表
* 构造函数可以是虚函数吗？不可以，因为对象中的虚表指针是在构造函数初始化列表阶段才初始化的
* 析构函数可以是虚函数吗？什么场景下析构函数是虚函数？可以，且最好把父类的虚表设置为虚函数，形成多态（见重写的例外情况），一个例子就是异常类的析构函数
* 拷贝构造和operator=可以是虚函数吗？拷贝构造不可以，拷贝构造也是构造有初始化列表；operator=可以，但是没有意义，因为赋值的参数需要是同类，这不能完成重写，但若是使用和父类拷贝构造相同的参数，又不能达到子类拷贝的目的，因此没有意义
* 对象访问普通函数快还是虚函数快？若不构成多态，就都是编译时决议，则一样快；若构成多态，则普通函数快，因为多态调用是运行时决议要通过虚表调用
* 虚函数表是在什么阶段生成的？存在哪里？虚表存在内存的常量区，在编译阶段就生成好了。注意不要和虚函数表的指针是在构造函数的初始化列表阶段生成的，此时去常量区找到虚表的地址并把它放到类对象里。

# map和set

## *pair 键值对*

```cpp
template <class T1, class T2>
struct pair
{
    typedef T1 first_type;
    typedef T2 second_type;
    T1 first;
    T2 second;
    pair(): first(T1()), second(T2()){}
    pair(const T1& a, const T2& b): first(a), second(b){}
};
```

* 将键值对打包成了一个结构体，这样便于解引用操作
* 用 `make_pair` 自动推导生成键值对

## *set和map的重要接口*

根据应用场景的不同，STL总共实现了两种不同结构的关联式容器：树形结构与哈希结构

树形结构的关联式容器主要有四种：map、set、multimap、multiset。这四种容器的共同点式使用平衡搜索树（即红黑树）作为其底层结构

### set

`lower_bound(val)` 返回的是 >= val的
`upper_bound(val)` 返回的是 > val的

* multiset：允许键值冗余

find返回的是中序第一个找到的key

erase删除的是所有的符合项

### map

* insert
  * `pair<iterator,bool> insert (const value_type& val)`
  * >The single element versions (1) return a pair, with its member pair::first set to an iterator pointing to either the newly inserted element or to the element with an equivalent key in the map. The pair::second element in the pair is set to true if a new element was inserted or false if an equivalent key already existed.
  * 返回类型是first为迭代器的pair是为operator[]准备的，否则就返回一个bool就可以了，没必要返回迭代器
* operator[]
  * 给一个key，若能找到符合的key，返回val&：相当于查找+修改val的功能
  * 若没有符合的，就插入默认构造生成的pair，即 `pair(key, V())`，并返回val&：相当于插入+修改的功能
  * 底层是利用insert实现的

    ```cpp
    V& operator[](const K& key)
    {
        pair<iterator, bool> ret = insert(make_pair(key, V()));
        return ret.first->second; //first是迭代器，对first解引用得到pair结构体，再取second
    }
    ```

* at相比于operator[]就只是查找+修改，若找不到就抛_out_of_range_exce的异常，这和python字典的功能相同
* multimap允许键值冗余，没有operator[]

## *set和map的模拟实现*

### 泛型编程

map和set的底层都是一棵泛型结构的RBTree，通过不同实例化参数，实现出map和set。对于map而言传的是`<Key, Pair<Key, Value>>`，对于set而言传的是`<Key, Key>`

```cpp
template<class K, class T, class KeyOfT>
struct RBTree {}

template<class K>
class set {//...
private:
    RBTree<K, K, SetKeyOfT> _t;
}

template<class K, class V>
class map {//...
private:
    RBTree<K, pair<K, V>, MapKeyOfT> _t;
}
```

### 节点的比较大小问题

对于set而言，`curr->_data` 可以直接比较，因为此时的data只是一个Key；但对于map而言，此时的data是一个pair键值对，需要用一个仿函数 KeyOfT 取出 pair 中的 Key 来实现大小比较

```cpp
struct SetKeyOfT
{
    const K& operator()(const K& key) {return key;}
};

struct MapKeyOfT
{
    const K& operator()(const pair<K, V>& kv) {return kv.first;}
};
```

### 迭代器

* set和map的迭代器和list迭代器的封装非常类似，也都使用了泛型编程。set和map的迭代器主要特殊点在于其 `operator++, operator--` 操作

* `begin()` 返回的是最左节点，即中序的第一个节点；因为迭代器的范围是左闭右开，所有`end()` 返回的是 `nullptr`

* `++` 返回的是中序下一个。因为是三叉链所以不需要借助辅助栈，可以直接找父亲

  * 若右子树不为空，`++` 就是找右子树中序下一个（最左节点）
  * 若右子树为空，`++` 找祖先里面当前孩子分支不是祖先的右的祖先。若找到了空的祖先，则说明走完了

  ```cpp
  Self& operator++() // 前置
  {
      if (_node->_right)
      {  // 右子树不为空就去找右子树的最左节点
          Node* left = _node->_right;
          while (left->_left)
              left = left->_left;
          _node = left;
      }
      else // 右子树为空就去找祖先里面当前孩子分支不是祖先的右的祖先
      {
          Node* parent = _node->_parent;
          Node* curr = _node;
          while (parent && curr == parent->_right) // parent为空就说明走完了
          {
              curr = curr->_parent;
              parent = curr->_parent;
          }
          _node = parent;
      }
      return *this;
  }
  ```

* `--` 返回的是中序前一个

  * 若左子树不为空，`--`就是找左子树中序上一个（最右节点）
  * 若左子树为空，`--`找祖先里面当前孩子分支不是祖先的左的祖先。若找到了空的祖先，则说明走完了

  ```cpp
  Self& operator--() {// 前置
      if (_node->_left) {
          Node* right = _node->_left;
          while (right->_right)
              right = right->_right;
          _node = right;
      }
      else {
          Node* parent = _node->_parent;
          Node* curr = _node;
          while (parent && curr == parent->_left) {
              curr = curr->_parent;
              parent = curr->_parent;
          }
          _node = parent;
      }
      return *this;
  }
  ```


##  *unordered_map/unordered_set*

* unordered系列容器是C++11新增的，其底层是Hash。在java中叫做 TreeMap/TreeSet & HashMap/HashSet
* 大量数据的增删查改用unordered系列效率更高，特别是查找，因此提供了该系列容器
* 与map/set的区别

  * map和set遍历是有序的，unordered系列是无序的
* map和set是双向迭代器，unordered系列是单向迭代器

### 用底层哈希桶进行封装

和用红黑树封装map和set一样，unordered_map和unordered_set的底层都是泛型结构的哈希桶，通过不同实例化参数，实现出unordered_map和unordered_set。对于unordered_map而言传的是`<Key, Pair<Key, Value>>`，对于unordered_set而言传的是`<Key, Key>`

### 迭代器

unordered系列是单向迭代器（哈希桶是单列表）

一个类型K去做set和unordered_set的模板参数的要求

* set要求能支持小于比较，或者显式提供比较的仿函数
* unordered_set
  * K类型对象可以转换整形取模或者提供转换成整形的仿函数
  * K类型对象可以支持等于比较，或者提供等于比较的仿函数。因为要找桶中的数据

# 右值引用 rvalue reference（C++11）

## *左值引用与右值引用*

### 左值 lvalue

对于早期C语言的，左值意味着

1. 它指定一个对象，所以引用内存中的地址
2. 它可用在复制运算符的左侧

**左值的特点是可以取地址**，**除了const**不能被赋值外其他可以放到赋值符号左边。因此为了适应const的变化，C标准新增了一个术语：**可修改的左值 modifiable lavalue**，用于标识可修改的对象。也可以称为对象定位值 object locator value

左值和右值的概念是相对于赋值操作而言的，而赋值的操作就是要把值存储到内存上

* 当一个对象被用作左值的时候，用的是对象的身份（在内存中的位置）
* 当一个对象被用作右值的时候，用的是对象的值（内容）

### 右值 rvalue

```cpp
//右值举例
10;
x + y;
fmin(x, y); //function
string("hello"); //匿名对象
```

右值不能出现在赋值符号的左边，且不能取地址。右值可以是常量、变量或其他可求值的表达式，比如函数调用

个人认为一个更好的理解方式是除常量外，若一个表达式能产生临时变量就可以认为这个表达式是右值，从函数的栈帧建立过程中我们就已经发现了这一点

### C++11中对右值的进一步划分

* 内置类型右值：纯右值 pure rvalue
* 自定义类型优质：将亡值 expiring value/xvalue

### 左值引用

左值引用只能引用左值不可以引用右值

但const左值可以引用右值，这样就不会因为左值改变而改变被引用的右值，因为const不能被改变

const左值可以引用右值这个特点在引用传参的时候被用到了，即引用传参时既可以接收左值也可以接收右值

```cpp
//左值引用可以引用右值吗：const的左值引用可以
//double& r1 = x + y; //错误
const double& r1 = x + y;
template<class T>
void Func(const T& T) {}
```

### 右值引用

```cpp
int&& rr1 = 10;
double&& rr2 = x + y;
double&& rr3 = fmin(x, y);
```

右值引用只能引用右值，不能引用左值

但是右值引用可以引用move以后的左值

需要注意的是右值是不能取地址的，但是给右值取别名后，编译器会给右值开一块空间，此时就可以对别名取地址了

```cpp
int&& rr1 = 10;
rr1 = 20; //合法
```

## *右值引用使用场景和意义*

==引用的核心价值是减少拷贝==

### 左值引用的短板

首先回顾一下左值引用的使用场景：做参数和做返回值[左值引用](#lvalue)

```cpp
string to_string(int val); //返回一个string临时对象，不能传引用返回，只能传值拷贝返回
void to_string(int val, string& str);

vector<vector<int>> generate(int numRows);
void generate(int numRows, vector<vector<int>>& w);
```

在上面的情境中，当要返回一个临时对象时，是不可以使用传引用返回的，因为栈帧被消灭了

考虑解决方案：全局变量会有线程安全问题，用new的话可能会有内存泄漏问题

但用输出型参数进行改造会又不太符合使用习惯，因为一般只要用外面一个变量接收一下就行了

### 右值引用和移动构造补齐短板

<img src="左值引用和右值引用返回对比.png">

右值引用的核心：在传值情况下通过移动构造（直接和要消亡的右值进行资源交换）减少深拷贝

右值引用不是像左值引用直接起作用的，而是通过识别右值来提供移动构造起作用的

C++11后的容器及其插入相关操作都支持了右值引用，主要就是解决了拷贝开销很大的问题，解决了传值返回这些类型对象的问题，比如push_back、insert

```cpp
vector<string> v;
string s1("hello");
v.push_back(s1); //左值插入，深拷贝
v.push_back(string("world")); //c++11支持了右值引用插入，匿名对象是一个右值
```

## *模板完美转发 Perfect forward*

### 万能引用

万能引用或引用折叠：模板中的 `&&` 不代表右值引用，而是万能引用。 既能引用左值（传左值时 `&&` 被折叠为 `&`），也能引用右值

但是最后都是统一成了左值引用

```cpp
void Fun(int& x) { cout << "左值引用" << endl; }
void Fun(const int& x) { cout << "const 左值引用" << endl; }

void Fun(int&& x) { cout << "右值引用" << endl; }
void Fun(const int&& x) { cout << "const 右值引用" << endl; }

// 万能引用/引用折叠：t既能引用左值，也能引用右值
template<typename T>
void PerfectForward(T&& t) {
	// 完美转发：保持t引用对象属性
	Fun(std::forward<T>(t));
}

int main() {
	PerfectForward(10);           // 右值
    
	int a;
	PerfectForward(a);            // 左值
	PerfectForward(std::move(a)); // 右值

	const int b = 8;
	PerfectForward(b);		      // const 左值
	PerfectForward(std::move(b)); // const 右值
   	return 0;
}
```

> 右值引用；左值引用；右值引用；左值引用；右值引用

### 完美转发

完美转换会保持引用对象的属性，比如list容器中支持了push_back，因为list是一个类模板，里面如果新增加push_back的右值引用版本，那么不论是左值还是右值都会走右值版本，因此用一个完美转发来区分

```cpp
template<class T>
class A {
	void push_back(const T& x) {
        insert(end(), x);
    }

    void push_back(T&& x) {
        insert(end(), std::forward<T>(x));
    }
};
```

### 右值引用带来的新的类默认成员函数

C++11之后默认成员函数变成了8个，增加了移动构造和移动赋值

* 只有在要实现深拷贝的时候才有显式实现这两个成员函数的价值，比如 string、vector、list
* 若不需要深拷贝，则可以自动生成，但自动生成的条件比较苛刻
  * 没有自己实现构造函数，且没有实现析构、拷贝构造、赋值重载中的任何一个（一般要自己实现析构就说明要清理资源的深拷贝，也就要同时实现拷贝和赋值重载），此时编译器才会自动生成一个默认移动构造
  * 默认移动构造会对内置类型按字节拷贝，即浅拷贝；对于自定义类型就要看它是否实现了移动构造，若实现了就调用移动构造，没有实现就用拷贝构造
  * 移动赋值和移动构造的条件和过程一样
  * 若显式提供了移动构造或移动赋值，那么编译器就不会提供拷贝构造和拷贝赋值

#	C++11 特性总结

## *统一的列表初始化*

列表初始化相当于直接调用了构造函数

```cpp
//支持初始化列表的构造函数
#include <initializer_list>
vector(initializer_list<T> il)
    :_start(nullptr)
    , _finish(nullptr)
    , _end_of_storage(nullptr) 
{
    reserve(il.size());
   	for (auto& e : il) {
    	push_back(e);
    }
}
```

map和pair都支持列表初始化

```cpp
map<string, string> dict = { { "sort", "排序" }, { "insert", "插入" } };
//auto dict{ { "sort", "排序" }, { "insert", "插入" } }; // 这样是错的，不知道里面自动推导的是pair
```

C++11以后一切对象都可以用列表初始化。但是建议普通对象还是用以前 `=` 赋值来初始化，容器如果有需求可以用列表初始化

## *处理类型*

### 类型别名 type alias

C语言提供了用typedef给类型起别名，从而简化一些特别长的自定义类型

C++11规定了一种新的方法，称为别名声明 alias declaration ，用关键字using来定义类型别名，比如

```c++
using iterator = _list_iterator<T, Ref, Ptr>;
```

但是给指针这种复合类型和常量起类型别名要小心一点，因为可能会产生一些意想不到的后果

### auto 变量类型自动推导和`decltype`

```cpp
int x = 10;
//typeid(x).name() y1= 20; //不能这么写
decltype(x) y1 = 20.22;
auto y2 = 20.22
```

* `auto` 可以进行自动变量推导
* `decltype`可以推导一个变量的类型，再用推导结果去定义一个新的变量

### `nullptr`

C语言中 NULL 被定义为常量0，因此在使用空指针时可能会出现一些错误。C++中引入了指针空值关键字 nullptr，它的类型是 `(void*)0`

## *一些新的关键字*

### 范围 for

范围for的底层就是直接替换迭代器实现，这在之前的STL容器的迭代器实现中已经说明了

### final 与 override

这两个关键字在继承与多态部分有使用过，链接：[C++ 11 override 和 final](#final)

### `default` 与 `delete`

* `default` 强制生成某个默认成员函数

* `delete` 禁止生成某个默认成员函数

  ```cpp
  //用delete关键字实现一个只能在堆上创建对象的类
  class HeapOnly {
  public:
  	//禁止析构生成，哪里都不能构造类对象
  	~HeapOnly() = delete;
  };
  
  int main() {
  	//自定义类型会调析构，指针不会
  	HeapOnly* ptr = new HeapOnly;
  	return 0;
  }
  ```

## *STL库的变化*

### 新增加容器：array、forward_list 以及 unordered 系列

增加array的初衷是为了替代C语言的数组，因为委员会认为C语言的数组由于其越界判定问题所以特别不好，数组的越界写是抽查，所以不一定能查出来；越界读除了常数区和代码区外基本检查不出来

array容器进行越界判定的方式是和vector类似的`[]`运算符重载函数。所以从越界检查的严格性和安全性角度来看还是有意义的。但因为用数组还是很方便而且用array不如用vector+resize。另一个问题是array和数组都是开在栈上，而vector是在栈上。所以C++11标准新增的array容器就变成了一个鸡肋

## *可变参数模板*

可变参数在C语言中就有了，比如printf的参数就是一个可变参数，底层是用一个数组来接收的。

C++11对可变参数进行了扩展，扩展到了模板中

### 模板声明

```cpp
template<class ...Args>
void showList(Args... args) {
	cout << sizeof...(args) << endl;
	////不能这么用
	//for (int i = 0; i < sizeof...(args); i++) {
	//	cout << args[i] << " ";
	//}
	//cout << endl;
}
```

### 递归函数方式展开参数包

```cpp
//0个参数的时候就不能递归调用原函数了，要补充一个只有val参数的函数重载，类似于递归的终结条件
void ShowList() {
	cout << endl;
}
//Args... args 代表N个参数包（N >= 0）
template<class T, class ...Args>
void ShowList(const T& val, Args... args) {
	cout << "ShowList(val：" << sizeof...(args) << " -- 参数包：";
	cout << val << "）" << endl;
	ShowList(args...);
}
```

### 逗号表达式展开参数包

## *lambda表达式*

### 像函数一样使用的对象/类型

* 函数指针
* 仿函数/函数对象
* lambda表达式函数**对象**

当类或函数模板需要用到仿函数时，此时可以用lambda表达式替换，因为这样可以将其隐藏到类或函数中封装

可以将lambda表达式理解为一个匿名函数表达式

### lambda表达式语法

<img src="lambdaexpsyntax.png">

```cpp
//两个数相加的lambda
auto add1 = [](int a, int b)->int {return a + b; };
cout << add1(1, 2) << endl;

//省略返回值
auto add2 = [](int a, int b){return a + b; };
cout << add2(2, 3) << endl;

//交换变量的lambda
int x = 0, y = 1;
//auto swap1 = [](int& x1, int& x2)->void {int tmp = x1; x1 = x2; x2 = tmp; }; //这样写很难看
//swap1(x, y);
//cout << x << " " << y << endl;

auto swap1 = [](int& x1, int& x2)->void {
    int tmp = x1;
    x1 = x2;
    x2 = tmp;
};
```

`[capture-list](parameters)mutable -> return-type {statement}` 没有函数名，记法：`[](){}`

* `[capture-list]` 捕捉列表：编译器根据 `[]` 来判断接下来的代码是否为lambda函数，捕捉列表能够捕捉父作用域的变量供lambda函数使用。本质还是==传参==
  * 捕捉方式
    * `[var]`：表示以值传递方式捕捉变量var，也就是说拷贝了var，对新var的改变不会改变原来的var
    * `[=]`：表示值传递的方式捕捉所有父作用域中的变量（包括this）
    * `[&var]`：表示引用传递捕捉变量var
    * `[&]`：表示引用传递捕捉所有父作用域中的变量（包括this）
  * 注意点
    * 父作用域指的是所有包含lambda函数的语句块
    * 允许混合捕捉，即语法上捕捉列表可由多个捕捉项组成，并以逗号分割，比如 `[=, &a]`，以值传递捕捉所以值，但对a采取引用捕捉
    * 捕捉列表不允许变量重复传递，否则会编译错误，比如 `[=, a]` 这种捕捉方式，已经以值传递方式捕捉了所有变量了，包括a
* `(parameters)` 参数列表：和普通函数的参数列表一致，若无参就可以和括号一同省略
* `mutable`：默认情况下，lambda函数总是一个const函数，mutable可以取消其常量性。使用该修饰符时，参数列表不可省略（也就是括号不能省略）
* `->return-type` 返回值类型：没有返回值或返回值类型明确情况下都可以省略，由编译器自动推导返回类型，因此lambda表达式在大多数情况下都不会写返回值类型
* `{statement}` 函数体

### lambda底层原理

和范围for底层是迭代器直接替换一样，lambda的底层就是仿函数类。在下图的汇编代码中可以看到，二者的汇编代码结构几乎完全相同

lambda函数对于用户是匿名的，但对于编译器是有名的，其名称就是lambda_uuid

<img src="lambda底层实现.png">

## *包装器 Wrapper*

### 包装器解决的2个问题

**可调用对象的种类过多引起的效率问题**

C++中可调用对象很多，`ret=func(x)` 中的 `func()` 既可以是函数指针，也可以是仿函数对象，还可以是lambda表达式

```cpp
template<class F, class T>
T useF(F f, T x) {};

// 函数名
cout << useF(f, 11.11) << endl;
cout << useF(f, 22.22) << endl;
// 函数对象
cout << useF(Functor(), 11.11) << endl;
// lamber表达式
cout << useF([](double d)->double{ return d/4; }, 11.11) << endl;
```

当有如上的用可调用对象作为模板参数的时候，如果直接这么写，即使传入的本质上参数和返回值完全相同的不同可调用对象，函数模板也会实例化多份。试验结果如下，当传入不同的可调用对象时，会生成不同的函数模板，前两个传入的都是函数地址，因此使用的是同一份模板。这回导致代码膨胀+效率降低

<img src="包装器试验.png">

**为可调用对象提供了一个统一的参数**

上面的 `useF(F f, T x) {};` 是一个函数模板，里面的参数F是一个为可调用对象设计的统一接口，可以往里面放任何的可调用对象。若没有包装器机制，就只能是调固定类型的函数指针、仿函数或者lambda函数了，最尴尬的是**当模板参数的类型是lambda可调用对象的时候，甚至连类型都写不了**

### function包装器

这个问题可以用function包装器来解决。包装器是一个**函数专用的类模板，其特点是统一类型**，减轻编译器自动推导的压力

```cpp
//包装器的头文件和使用方法
#incldue <fucntional>
template <class Ret, class... Args>
class function<Ret(Args...)>; //Ret 为被调用函数的返回类型， Args 为调用函数的参数包
```

具体到上面的情况为如下

```cpp
// def for f, Functor()
std::function<double(double)> func1 = f;
cout << useF(func1, 11.11) << endl;
// 函数对象
std::function<double(double)> func2 = Functor();
cout << useF(func2, 11.11) << endl;
// lamber表达式
std::function<double(double)> func3 = [](double d)->double { return d/4; };
cout << useF(func3, 11.11) << endl;
```

此时就可以统一使用一个函数模板了

<img src="包装器试验2.png">

介绍一个应用：[150. 逆波兰表达式求值 - 力扣（LeetCode）](https://leetcode.cn/problems/evaluate-reverse-polish-notation/)

### 类成员函数的包装问题

静态成员函数可以直接包装，因为它的参数没有多this指针。但是对于普通函数参数中有一个多出来的this指针需要特殊处理

* 在包装的时候，类非静态成员函数的包装器要多一个不可省略的参数 `Plus`（C++11规定了传类名，而不是 `this`），并且还要非静态成员函数的地址，也是C++的规定
* 并且若采用函数名调用，需要在参数中添加一个匿名对象，比如 `func5(Plus(), 11.11, 11.11);`

```cpp
class Plus {
public:
	static int plusi(int a, int b) {
		return a + b;
	}
	double plusd(double a, double b) {
		return a + b;
	}
};
//调用包装器
std::function<int(int, int)> func4 = Plus::plusi;
cout << useF(func4, 11.11) << endl;
std::function<double(Plus, double, double)> func5 = &Plus::plusd; //非静态成员函数要取地址，C++规定, Plus相当于是this指针
cout << useF(func5, 11.11) << endl;
func5(Plus(), 11.11, 11.11); //直接调用要传入匿名对象
```

### `std::bind()` 解决参数数量、顺序不匹配的问题

在下面这种情况时，因为map数据结构的类型需要的输入包装器已经写死了是要以int为返回值，以两个int为输入，此时无法匹配map带有三个参数的非静态类成员函数的function包装器

```cpp
map<string, std::function<int(int, int)>> opFuncMap = { //第二个模板参数必须要传入两个参数的
    {"普通函数指针", f},
    {"函数对象", Functor() },
    {"成员函数指针", &Plus::plusi} //报错
}
```

此时可以用 `std::bind()` 通用函数适配器来**调整可调用对象的参数个数和顺序**。`std::bind` 函数是一个函数模板，它会生成一个新的可调用对象

```cpp
class Plus {
public:
	Plus(int x = 2)
		:_x(x)
	{}
	int plusi(int a, int b) {
		return (a + b)*_x;
	}
private:
	int _x;
};

int main() {
	std::function<int(Plus, int, int)> func3 = &Plus::plusi;
	cout << func3(Plus(), 100, 200) << endl;

	std::function<int(int, int)> func4 = std::bind(&Plus::plusi, Plus(10), \
		std::placeholders::_1, std::placeholders::_2); //绑定参数，std::placeholder是占位符
   
    //调整顺序
	cout << func4(100, 200) << endl;
	return 0;
}
```

`std::placeholder::_1` 是一个占位符，还剩下几个绑定后的参数就用几个占位符

* 调整参数个数，一般是实际要传入的参数个数比接受的传入参数个数要多

  ```cpp
  //绑定两个参数
  std::function<int(int, int)> func4 = std::bind(&Plus::plusi, Plus(10), \
  	100, std::placeholders::_1); //绑定参数，std::placeholder是占位符
  ```

* 调整参数顺序

  ```cpp
  	std::function<int(int, int)> func4 = std::bind(&Plus::plusi, Plus(10), \
  		std::placeholders::_2, std::placeholders::_1); //绑定参数，std::placeholder是占位符
  ```

**绑定参数的原理是相当于先把某些参数传进去了，然后返回一个已经传入部分参数的函数继续接收参数**

### 包装器在TCP server 派发任务中的应用

看计算机网络套接字编程 TCP server 部分

## *线程库*

# 异常 Exception

## *C++异常概念*

### C语言中的错误码

C语言采用的是传统的错误处理机制

* 终止程序，如 `assret`，缺陷是比较粗暴，难以排查，因为只有很严重的错误才应该直接终止程序
* 返回错误码，如 `perror` 等，缺陷是需要程序员自己根据错误码来排查对应的错误

### C++异常概念

异常是一种处理错误的方式，当一个函数发现自己无法处理的错误时就可以抛异常，让函数的直接或间接地调用者来处理这个错误

异常体系的三个关键字

* `throw`：当问题出现时，程序会抛出一个异常，则是通过使用 `throw` 关键字来完成的
* `catch`：在用户想要处理问题的地方，通过 `catch` 来捕获异常，可以有多个 `catch`
* `try`：`try` 块中的代码标识符将被激活的特定异常，它后面通常跟着一个或多个 `catch` 块。`try` 块中的代码被称为保护代码

```cpp
while (1) {
    try {
        Func();//保护代码里写throw来处理错误
    }
    catch (const char* errmsg) {
        //...
    }
    catch (int msgNO) {
        //...
    }
    //...
    catch (...) { //捕获任意类型的异常，防止出现未捕获异常时，程序直接终止
        cout << "Unknown exception" << endl;
    }
}
```

## *异常的使用*

### 异常的抛出和匹配原则

* 异常是通过抛出对象而引发的，该对象的类型决定了应该激活哪个catch的处理代码
* 被选中的处理代码是**调用链 call chain**中与该对象类型匹配且离抛出异常位置最近的那一个
* 抛出异常对象后，会生成一个异常对象的拷贝，因为抛出的异常对象可能是一个局部临时对象，类似于传值返回
* `catch (...)` 可以捕获任意类型的异常，问题是不知道异常错误是什么。这是C++中**保证程序健壮性的最后一道底线**，必须要写
* 实际中抛出和捕获的匹配原则有个例外，并不都是类型完全匹配，可以抛出子类对象，使用父类捕获（切片）

### 在函数调用链中异常栈展开匹配原则

<img src="callChain.png">

* 首先检查 `throw` 本身是否在 `try` 块内部，如果是的话就再查找匹配的 `catch` 语句，若有匹配的就跳到catch的地方进行处理
* 没有匹配的 `catch` 则退出当前函数栈，继续在调用函数的栈中进行查找匹配的 `catch`
* 若达到main函数栈依旧没有匹配的 `catch` 就报错
* 找到匹配的 `catch` 字句并处理以后，会继续沿着 `catch` 字句后面继续处理

### 异常的重新抛出

有可能单个的catch不能完全处理一个异常，在进行一些校正处理以后，希望再交给更外层的调用链函数来处理，catch则可以通过重新抛出将异常传递给更上层的函数进行处理

```cpp
void Func() {
	// 这里可以看到如果发生除0错误抛出异常，另外下面的array没有得到释放
	// 所以这里捕获异常后并不处理异常，异常还是交给外面处理，这里捕获了再重新抛出去
	int* array = new int[10];
	int len, time;
	cin >> len >> time;

	try {
		cout << Division(len, time) << endl;
	}
	catch (...) 
	{
		cout << "delete []" << array << endl;
		delete[] array;

		throw; // 捕获什么抛出什么
	}
	cout << "delete []" << array2 << endl;
	delete[] array2;
}
```

### 异常安全

* 构造函数完成对象的构造和初始化，最好不要在构造函数中抛出异常，否则可能导致对象不完整或没有完全初始化

* 析构函数主要完成资源的清理，最好不要在析构函数中抛出异常，否则可能导致资源泄露

* C++中异常经常会导致资源泄漏的问题，比如在 `new` 和 `delete` 中抛出了异常，导致内存泄漏（如下面的例子），且处理这种情况很麻烦。或者在 `lock` 和 `unlock` 之间抛出了异常导致死锁。C++经常使用RAII来解决以上问题

  ```cpp
  void Func() {
  	// 1、如果p1这里new 抛异常会如何？
  	// 2、如果p2这里new 抛异常会如何？
  	// 3、如果div调用这里又会抛异常会如何？
  	int* p1 = new int;
  	int* p2 = new int;
  
  	cout << div() << endl;
  
  	delete p1;
  	delete p2;
  	cout << "释放资源" << endl;
  }
  ```

### 异常规范

异常规范是一种最好遵守的建议，但它不能做到强制程序员遵守，因为C++需要兼容C语言，而C语言中并没有异常体系

C++11中新增关键字 `noexcept`，表示不会抛异常

## *异常体系*

### 自定义异常体系

<img src="自定义异常体系.png">

### C++异常体系

<img src="CppExceptionSystem.png" width="45%">

注意一个点，除零错误不是C++的标准异常，因此如果不throw来try除零的代码，不会抛异常

## *异常的优缺点*

### 优点

* 比起错误码而言可以展示更丰富的信息，甚至可以包含堆栈调用的信息，帮助用户更好地定位程序bug
* 调用链很深的情况下，可以直接抛异常给外层接受处，不需要层层返回
* 很多的第三方库都包含异常，比如 `boost, gtest, gmock` 等等常用的库，使用它们的时候也需要使用异常
* 部分函数使用异常更好处理，比如构造函数没有返回值，不方便使用错误码方式处理，比如越界使用异常或者直接 `assert` 终止程序

### 缺点

* 导致程序地执行流乱跳，非常混乱，有点像 `goto`。程序的运行有时候往往超乎用户想象，此时用比如打断点的方式可能就不能很好的调试程序。这个缺点是最严重的，其他缺点都或多或少有解决方法
* 异常要拷贝对象，有一些多余的性能开销，但这个问题随着硬件发展已经几乎可以忽略
* C++没有垃圾回收机制 Garbage Collection GC，需要用户自己管理资源。有了异常就非常容易造成内存泄漏、死锁等异常安全问题。这个需要使用RAII来处理资源的管理问题
* C++标准库的异常体系定义的不好，导致不同公司、不同项目之间会自定义各自的异常体系，非常混乱
* 虽然C++有异常规范，但由于各种历史原因，规范不是强制的。异常要尽量规范使用，否则会造成严重后果

### 总结

异常总体而言，利大于弊，所以在大型工程中还是要鼓励使用异常，而且基本所有的面向对象语言都用异常来处理错误

# 智能指针 Smart Pointer（C++11）

后来发展的面向对象语言因为借鉴了C++缺乏有效资源管理的机制，都发展除了垃圾回收机制。智能指针是C++为了补不设置垃圾回收机制的坑，且垃圾回收对于主程序而言是一个独立的进程，会有一定的性能消耗，C++考虑到性能也就没有采取垃圾回收的方法

但智能指针主要是为了==保证异常安全==

## *智能指针的使用及原理*

### RAII思想

RAII Resource Acquisition Is Initialization 资源获取即初始化 是一种利用对象生命周期来控制程序资源（如内存、文件句柄、网络接连、互斥量等等）的技术。Java中也会利用这种思想，虽然Java有垃圾回收机制，但同样会面对加锁和解锁时内存资源没有正常释放的问题

在对象构造时获取资源，最后在对象析构的时候析构资源，不论在任何情况下当对象退出所在的内存空间，也就是说其生命周期结束后，一定会调用析构进行清理，这是由语法定义决定的。相当于把管理一份资源的责任托管给了一个对象。这样做有两大好处

* 不需要显式地释放资源
* 采用这种方式，对象所需的资源在其生命周期内始终保持有效

### 原理与模拟实现

实现主要有3个方面

* RAII行为
* 支持指针操作
* 拷贝析构问题

```cpp
//利用RAII设计delete资源的类
template<class T>
class SmartPtr {
public:
	SmartPtr(T* ptr)
		:_ptr(ptr)
	{}
	
	~SmartPtr() {
		cout << "delete: " << _ptr << endl;
		delete _ptr;
	}
    //要支持指针操作行为
    T& operator*() {
		return *_ptr;
	}

	T* operator->() {
		return _ptr;
	}
    
private:
	T* _ptr;
};

void Func() {
	SmartPtr<int> sp1(new int); //若这里new出问题抛异常，那么退出后由类对象析构进行处理
	SmartPtr<int> sp2(new int); //若这里new出问题抛异常，那么sp1和sp2也会调用析构处理
    cout << div() << endl; //若这里出问题，也是一样的
}
```

## *C++标准库提供的智能指针*

### C++98: `std::auto_ptr`

上面实现的“智能指针”有浅拷贝问题：和迭代器的行为非常类似，都是故意要浅拷贝，因为我们想要用拷贝的指针去管理同一份资源，但是对上面实现的智能指针就会出现浅拷贝析构问题。而迭代器浅拷贝析构不会报错的原因是因为轮不到迭代器进行析构，迭代器只是封装，容器会将所有的内容一块析构掉

对此 `std::auto_pair` 的解决方法是管理权转移，这是一种极为糟糕的处理方式，类似对左值进行了右值处理，直接交换了原指针的资源。会导致被拷贝对象悬空，再次进行解引用就会出现对空指针解引用问题。因此绝大部分公司都明确禁止使用这个指针类来进行资源管理

```cpp
//管理权转移的实现
auto_ptr(auto_ptr<T>& sp)
:_ptr(sp._ptr) {
	sp._ptr = nullptr;
}

auto_ptr<T>& operator=(auto_ptr<T>& ap) {
	// 检测是否为自己给自己赋值
	if (this != &ap) {
		// 释放当前对象中资源
		if (_ptr)
			delete _ptr;
		// 转移ap中资源到当前对象中
		_ptr = ap._ptr;
		ap._ptr = NULL;
	}
	return *this;
}
```

### C++11: `std::unique_ptr`

C++11 的 `std::unique_ptr` 是从先行者boost库中吸收过来的，原型是 `scoped_ptr`

```cpp
//C++98只能通过声明而不实现+声明为私有的方式来做，但C++11可以用delete关键字
unique_ptr(unique_ptr<T>& ap) = delete; //禁止生成默认拷贝构造
unique_ptr<T>& operator=(unique_ptr<T>& ap) = delete; //禁止生成默认赋值重载
```

非常的简单粗暴，直接禁止了拷贝构造，但并没有从根本上解决问题。只适用于一些不需要拷贝的场景

### C++11: `std::shared_ptr` 的引用计数机制

==`std::shared_ptr` 是智能指针和面试中的重点==

采取和进程PCB块中的程序计数一样的思想，即引用计数：每个对象释放时，--计数，最后一个析构对象时，释放资源

利用静态成员变量实现是不对的，因为静态变量是属于类的所有对象的，因此在有多个类时会共享一个计数器，这就起不到计数的作用了

<img src="sharedPtr的引用计数机制.png" width="80%">

```cpp
template<class T, class D = Delete<T>>
class shared_ptr 
public:
shared_ptr(T* ptr = nullptr)
    : _ptr(ptr)
        , _pCount(new int(1)) //给一个计数器
    {}

~shared_ptr() {
    Release();
}

void Release() {
    if (--(*_pCount) == 0) { //给对象赋值是建立在*this目标已经定义的情况下的
        // 此时计数器至少为1，若没有这步，直接更改指向对象会造成内存泄漏
        cout << "Delete: " << _ptr << endl;
        //delete _ptr;
        D()(_ptr);
        delete _pCount;
    }
}

shared_ptr(shared_ptr<T>& sp)
    :_ptr(sp._ptr)
        , _pCount(sp._pCount)
    {
        (*_pCount)++;
    }

shared_ptr<T>& operator=(const shared_ptr<T>& sp) {
    //防止自己给自己赋值
    if (_ptr == sp._ptr) {
        return *this;
    }

    Release();

    _ptr = sp._ptr;
    _pCount = sp._pCount;

    (*_pCount)++;
    return *this;
}

T& operator*() {
    return *_ptr;
}

T* operator->() {
    return _ptr;
}

T* get() { //给weak_ptr使用
    return _ptr;
}

private:
T* _ptr;
int* _pCount; //计数器
D _del;
};
```

### `std::shared_ptr` 的线程安全

### `std::shared_ptr` 的循环引用问题

<img src="智能指针循环引用.png">

如上图所示，当退出 `test_shared_ptr2()` 时，n1和n2指针虽然销毁了，但new出来的空间还在，分别被右边的 `_prev` 和左边的 `_next` 管理，此时两个计数器都回到1。然后就产生了一个逻辑矛盾的销毁路径。这个问题被称为循环引用 circular reference

该问题用 `std::weak_ptr` 来解决， `std::weak_ptr` 不是常规智能指针，没有RAII，也不支持直接管理资源

 `std::weak_ptr` 主要用 `std::shared_ptr` 来构造，因此不会增加计数，==本质就是不参与资源管理==，但是可以访问和修改资源

```cpp
template<class T>
class weak_ptr { //自己实现，库里的比这个复杂得多
public:
    weak_ptr()
        :_ptr(nullptr)
        {}

    weak_ptr(const shared_ptr<T>& sp) //支持对shared_ptr的拷贝构造
        :_ptr(sp.get())
        {}

    weak_ptr(const weak_ptr<T>& wp)
        :_ptr(wp._ptr)
        {}
}
```

以下是利用 `std::weak_ptr` 解决循环引用问题

```cpp
struct Node {
    int _val;
    std::weak_ptr<Node> _next;//解决循环引用，不会增加计数
    std::weak_ptr<Node> _prev;

    ~Node() {
        cout << "~Node()" << endl;
    }
};

//循环引用，没有报错是因为main退出后会自动清理资源
//但很多程序是需要长时间运行的，在这种情况下的内存泄漏是很可怕的
void test_shared_ptr2() {
    std::shared_ptr<Node> n1(new Node);
    std::shared_ptr<Node> n2(new Node);
    n1->_next = n2;
    n2->_prev = n1;
}
```

### 定制删除器

如上面自己实现的 `shared_ptr` 所示，析构的时候其实不知道到底该用 `delete` 或 `delete[]`，甚至有可能数据实用malloc出来的，为了规范，此时应该要用free。特别是 `[]` 问题，不匹配的结果是很可怕的

因此就要给一个模板，显式传入要用哪种delete方式

`std::shared_ptr` 不是实现的类模板，而是实现了构造函数的函数模板，这样比较直观也符合逻辑，但这种实现方式是很复杂的，和上面我们自己实现的用类模板不一样

下面给出两个仿函数的例子

```cpp
template<class T>
struct DeleteArray {
    void operator()(T* ptr) {
        cout << "delete" << ptr << endl;
        delete[] ptr;
    }
};

template<class T>
struct Free {
    void operator()(T* ptr) {
        cout << "free" << ptr << endl;
        free(ptr);
    }
};

//调用仿函数对象
std::shared_ptr<Node> n1(new Node[5], DeleteArray<Node>());
std::shared_ptr<Node> n2(new Node);
std::shared_ptr<int> n3(new int[5], DeleteArray<int>());
std::shared_ptr<int> n4((int*)malloc(sizeof(12)), Free<int>());
```

但大部分情况下，都会直接使用lambda来传

```cpp
//lambda
std::shared_ptr<Node> n1(new Node[5], [](Node* ptr) {delete[] ptr; });
std::shared_ptr<Node> n2(new Node);
std::shared_ptr<int> n3(new int[5], [](int* ptr) {delete[] ptr; });
std::shared_ptr<int> n4((int*)malloc(sizeof(12)), [](int* ptr) {free(ptr); });
```