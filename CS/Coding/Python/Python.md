[Python最佳实践指南！ — The Hitchhiker's Guide to Python (pythonguidecn.readthedocs.io)](https://pythonguidecn.readthedocs.io/zh/latest/)



# Python 虚拟环境 & 包管理器

## *安装 Python*

Mac 用 Brew install，Linux 用相应的包管理器就可以了

### Win 安装 Python

[Python Releases for Windows | Python.org](https://www.python.org/downloads/windows/)

特别要注意勾上`Add Python 3.x to PATH`，然后点“Install Now”即可完成安装

### py 启动器

另外在 Win terminal 中是用 `py` 启动器命令启动 Python，这是一个由 Python 官方为 Windows 用户提供的工具。而不是像 Mac 或 Linux 的 `python` 或 `python3`

1. Python 启动器 `py` 允许在系统上安装有多个 Python 版本时，轻松地选择要使用的版本。可以通过像 `py -2` 或 `py -3` 这样的命令来分别启动 Python 2.x 或 Python 3.x。还可以使用更具体的版本号，如 `py -3.7` 来启动 Python 3.7
2. **避免冲突**：Windows 系统中通常不会将 `python` 或 `python3` 作为命令添加到环境变量 PATH 中。原因之一是为了避免与其他软件或系统自带的脚本发生命名冲突。例如，在某些版本的 Windows 上，当 `python` 命令不存在时，执行 `python` 命令可能会被重定向到 Microsoft Store 来提示用户安装 Python

## *pip*

pip 是 Python 的包管理工具，用于安装和管理 Python 包（也称为模块或库）。它是 Python Package Index（PyPI）上的软件包仓库的客户端工具，允许用户轻松地下载、安装、升级和卸载 Python 包

pip需要结合virtualenv或vene虚拟环境管理工具一起使用

### **安装pip**

* 在 Python 3.4 及更高版本中，`pip` 已经内置，无需额外安装

* 在较早的 Python 版本中，可能需要手动安装 pip

  ```cmd
  python -m ensurepip --default-pip
  ```

在用户级别进行安装，而不是系统级别，通常使用 `--user` 选项。这样可以避免需要管理员权限

### **基本命令**

* `pip install <package>`：安装指定的 Python 包

  * pip 能够自动解析和安装包的依赖项，确保所有必需的库和模块都被正确安装

  * pip 可以进行版本控制，即允许用户指定要安装的包的特定版本，以确保项目的稳定性

    ```cmd
    pip install SomePackage==1.0.4
    ```

* `pip uninstall <package>`：卸载指定的 Python 包

* `pip freeze`：列出当前环境中安装的所有包及其版本

* `pip list`：以更简洁的格式列出已安装的包

* `pip show <package>`：显示有关指定包的详细信息

* `pip search <keyword>`：搜索 PyPI 上的包

## *virtualenv & venv*

考虑到虚拟环境的重要性，Python 从3.3 版本开始，自带了一个虚拟环境模块 [venv](https://cloud.tencent.com/developer/tools/blog-entry?target=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fvenv.html&source=article&objectId=2124483)，关于该模块的详细介绍，可参考 [PEP-405](https://cloud.tencent.com/developer/tools/blog-entry?target=http%3A%2F%2Flegacy.python.org%2Fdev%2Fpeps%2Fpep-0405%2F&source=article&objectId=2124483) 。它的很多操作都和 virtualenv 类似。如果使用的是python3.3之前版本或者是python2，则不能使用该功能，依赖需要利用virtualenv进行虚拟环境管理

### virtualenv

virtualenv 是一个创建隔绝的Python环境的 工具。virtualenv创建一个包含所有必要的可执行文件的文件夹，用来使用Python工程所需的包

```cmd
pip install virtualenv
```

### venv

## *conda*

Conda是一个用于管理和部署软件包的开源包管理工具和环境管理器，Conda可以帮助用户创建、管理和切换不同的Python环境，并安装各种软件包，使得项目之间的依赖关系更加清晰和可管理

安装conda [Miniconda — Anaconda documentation](https://docs.anaconda.com/free/miniconda/)

```c,d
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

用zsh安装sh可能会有问题，这时候可以换用bash安装

```cmd
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

### 使用

* 创建和管理环境

  * 使用Conda可以轻松创建新的Python环境，例如：

    ```cmd
    $ conda create --name myenv python=3.8
    ```

    这将创建一个名为 "myenv" 的新环境，并指定Python版本为3.8

  * 激活/切换环境：

    ```cmd
    $ conda activate myenv
    ```

    激活环境后可以在其中安装软件包，运行Python脚本等

  * 退出环境：

    ```cmd
    $ conda deactivate
    ```

  * 删除环境

    ```cmd
    $ conda remove --name ENV_NAME --all
    ```

* 安装和管理软件包

  * 使用Conda可以轻松安装、更新和删除软件包，例如：

    ```cmd
    $ conda install numpy
    ```

    这将安装名为 "numpy" 的Python包

  * 更新软件包：

    ```cmd
    $ conda update numpy
    ```

  * 删除软件包：

    ```cmd
    $ conda remove numpy
    ```

* 创建环境文件

  * 可以通过创建一个环境文件（例如environment.yml）来定义项目的环境依赖关系。这个文件可以包含项目所需的所有软件包及其版本信息

  * 通过以下方式创建环境：

    ```cmd
    $ conda env export > environment.yml
    ```

* 从环境文件创建环境

  * 若有一个环境文件，可以使用以下命令从文件中创建一个新的环境

    ```cmd
    $ conda env create -f environment.yml
    ```

* 查看已安装的环境和软件包

  * 查看所有已创建的环境

    ```
    $ conda info --envs
    ```

  * 查看当前激活的环境中安装的软件包

    ```cmd
    $ conda list
    ```

### 仓库

`conda` 使用它自己的软件仓库，称为 Anaconda Repository。此外，`conda` 还可以使用其他仓库，例如 conda-forge

## *集大成之作：Pipenv*

pipenv 是Kenneth Reitz（requests的作者）大神的作品。它结合了 Pipfile，pip，和virtualenv，能够有效管理Python多个环境，各种包

https://cloud.tencent.com/developer/article/2124483

https://www.cnblogs.com/zingp/p/8525138.html

```cmd
$ pip install pipenv
```

pipenv创建虚拟环境后，会在工程目录生成如下两个文件：

- Pipfile：用于保存项目的python版本、依赖包等相关信息。该文件是可移植的，可以被单独移放到其他项目内，用于项目虚拟环境的建立和依赖包的安装
- Pipfile.lock：用于对Pipfile的锁定

## *模块 Module*

### 命名空间

和 C++ 使用 `{}` 来显式定义命名空间 namespace 不同，Python中每一个 `.py` 文件都是一个独立的模块，即一个独立的命名空间

通过使用模块，可以有效地避免命名空间的冲突，可以隐藏代码细节让我们专注于高层的逻辑，还可以将一个较大的程序分为多个文件，提升代码的可维护性和可重用性

Python还可以导入并利用其他语言的代码库，如C/C++的动、静态库

一共有三种命名空间，查找变量的时候会按照 局部的命名空间 -> 全局命名空间 -> 内置命名空间 的顺序去寻找

* **内置名称 built-in names**：Python 语言内置的名称，比如函数名 abs、char 和异常名称 BaseException、Exception 等等
* **全局名称 global names**：模块中定义的名称，记录了模块的变量，包括函数、类、其它导入的模块、模块级的变量和常量
* **局部名称local names**：函数中定义的名称，记录了函数的变量，包括函数的参数和局部定义的变量。（类中定义的也是）

### 撰写模块的说明文档

在实际工程中要为每个模块、每个类和每个函数都要撰写说明文档

```python
'''
Documentation
'''
import demo
print(demo.__doc__) #查看文档
```

### 导入模块

* 推荐的做法：导入整个模块 `import demo`，导入demo模块中的所有内容，但**会保留其命名空间**，需要用 `demo.成员` 指定模块名的方式来进行调用
* 导入模块中的特定部分
  * `from demo import 成员名`：从demo模块中导入指定的成员，会将该成员从原来的命名空间中**合并**到目前的命名空间中，因此不需要 `demo.` 就可以直接调用
  * 尽量不要使用 `from demo import *`，很容易出现重复定义的情况
* 使用 `as` 指定别名 `import numpy as np`，此时也需要通过 `.` 访问符来访问

### 测试单元

Python不需要一个类似C++工程中main函数的程序入口，任何 `.py` 文件都可以是一个单独的程序，解释器会自动逐行执行。但是这也会执行包含import的模块中的程序块

为了不让import的程序块执行，可以将单元测试脚本写在 `if __name__ == '__main__':` 中，比如

```python
def test_func():
   pass

if __name__ == '__main__':
    #Test Code for this .py file
```

### 模块路径

对于用import语句导入的模块，Python会按照下面的路径按顺序来查找

1. 当前的工作目录
2. `PYTHONPATH` 环境变量中的每一个目录
3. Python默认的安装目录：UNIX下，默认路径一般为 `/usr/local/lib/python/`

## *包 Package*

Python包，就是里面装了.py文件的文件夹

### 包的性质

* 包本质上是一个文件夹
* 该文件夹里一定有一个 `__init__.py` 模块
* 包的本质依然是模块，因此一个包里面还可以装其他的包

### `__init__.py` 文件

* 本身是一个模块
* 这个模块的模块名不是 `__init__`，而是这个包的名字，也就是装着 `__init__.py` 文件的文件夹名字
* 它的作用是将一个文件夹变为一个Python模块
* `__init__.py` 中可以不写代码，但是此时仅仅是 `import pkg` 的话就什么都不会发生，一般会重写 `__all__` 方法来规定当imprt该包的时候会自动import包中的哪些模块
* 不建议在 `__init__.py` 中写类，以保证该py文件的纯净

### 导入包

和导入模块相同

### 关于库的问题

严格来说Python中是没有库 library的，模块和包都是Python语法概念，而库只是一个通俗的说法，平时说的库在Python中的具体化表现既可以是一个包也可以是一个模块

可以看到Python标准库中就是一堆的,py文件模块和文件夹（包）

## *requirements.txt*

`requirements.txt` 是一个文本文件，用于列出Python项目的依赖包以及它们的版本号，以确保在不同环境中都能准确地重建相同的Python环境

* 构建

  * 手动构建

    ```
    package1==x.x.x
    package2>=y.y.y
    ```

  * 自动生成

    ```cmd
    $ pip freeze > requirements.txt
    ```

* 使用

  ```cmd
  $ pip install -r requirements.txt
  ```

# Object

下面参考的是Cpython 3.7的代码

```cmd
$ git clone --branch v3.7.0 --depth 1 https://github.com/python/cpython.git
```



## *对象、变量和引用*

https://flaggo.github.io/python3-source-code-analysis/

### 一切皆对象

Python 中所有类的父类为 object 类（当然从下文再深究下去其实不是，为方便理解可以这么说），即任何变量、函数、类、数据结构全部都可以看做是一个object的派生类，下面给出Cpython中的object类

```c++
typedef struct _object {
    int ob_refcnt;               
    struct _typeobject *ob_type;
    //int ob_size; //记录变长对象的长度
} PyObject;
```

可以发现， Python 对象的核心在于一个引用计数和一个类型信息

* 其中 `ob_refcnt` 记录对象的引用数，当有新的指针指向某对象时，ob_refcnt 的值加 1， 当指向某对象的指针删除时，ob_refcnt 的值减 1，当其值为零的时候，则可以将该对象从堆中删除

* `_typeobject` 类型的指针 `ob_type`。这个结构体用于表示对象类型，具体记录了对象的性质

  ```c++
  typedef struct _typeobject {
      PyObject_VAR_HEAD
      const char *tp_name; /* For printing, in format "<module>.<name>" */
      Py_ssize_t tp_basicsize, tp_itemsize; /* For allocation */
  
      // ...... 省略部分暂时不关心的内容
  } PyTypeObject;
  ```

  * `ty_name`：类型名
  * `tp_basicsize, tp_itemsize`：创建类型对象时分配的内存大小信息
  * 被省略掉的部分：与该类型关联的操作（函数指针）
  * `PyObject_VAR_HEAD` 是另一个 `PyTypeObject` 类型的对象，封装了 `PyType_Type`，**是真正意义上的所有class的基类**，被称作 metaclass

* 对于变长对象，还会再增加一个 `ob_size` 来记录变长对象的长度


### Object的分类

<img src="object-category.jpg" width="60%">

- Fundamental 对象：类型对象
- Numeric 对象：数值对象
- Sequence 对象：容纳其他对象的序列集合对象
- Mapping 对象：类似 C++中的 map 的关联对象
- Internal 对象：Python 虚拟机在运行时内部使用的对象

### `isinstance()` &  `type()`

[python isinstance()方法的使用 - 志不坚者智不达 - 博客园 (cnblogs.com)](https://www.cnblogs.com/linwenbin/p/10491669.html)

* isinstance 会认为子类是一种父类类型，考虑继承关系
* type 不会认为子类是一种父类类型，不考虑继承关系

## *Python 中特殊的引用*

### 变量及引用

https://www.jianshu.com/p/5d8ec56b6d14

* Python是一种动态特性语言，变量不需要显式声明类型，对象的类型是内置的。解释器会自动推断变量的类型

* Python中一切皆对象，**变量是对象或者说是对象在内存中的引用**。因此变量类型指的是变量所引用的对象的类型

  ```python
  In [1]: a = 1
  
  In [2]: id(a)
  Out[2]: 140701612513584
  
  In [3]: a = 3
  
  In [4]: id(a)
  Out[4]: 140701612513648
  ```

  因此会发生各种现象，这在 C++ 中是不可想象的，因为 C++ 中变量 a 是绑定了内存空间的，问题只是往里面放的 int是什么而已；但在 Python 中对象才是绑定了内存空间，变量则是自由的，可以随便更改引用

* **C++ 中用等号来赋值或开辟内存空间，而 Python 中用等号来创建对对象的引用**，这是两门语言最大的不同（当然本质是因为Python中一切皆对象），为了方便理解，笔者仍然将等号称作赋值而不是引用

* 变量在内存中不占有空间，因为它是实际对象的引用。这和 C++ 中的引用概念是一样的

* 每个变量在使用前都必须用 `=` 赋值，变量赋值以后才会被创建

通过下面两个例子来理解变量与引用的关系：

```python
a = 'Jack'
b = a
a = 'Tom'
```

上面的代码执行过程为

1. 在内存中创建一个字符串对象 `'Jack'`，然后创建一个变量 a 来引用该字符串对象，此时字符串对象的引用计数器 `ob_refcntc + 1`
2. 在内存中创建一个变量 b，并把它作为 a，也就是字符串对象 `'Jack'` 的另外一个引用，此时字符串对象的引用计数器 `ob_refcntc + 1 == 2`
3. 在内存中创建一个字符串对象 `'Tom'`，然后将现有的变量 a 来引用该字符串对象，此时字符串对象 `'Jack'` 的引用计数器 `ob_refcntc - 1 == 1`，而字符串对象 `'Tom'` 的引用计数器 `ob_refcntc + 1 == 1`

```python
a = 1
a = "hello"
a = [1, 2, 3]
```

上面的代码执行过程为

1. 创建一个变量 a 来引用数值池中的对象 1
2. 在内存中创建一个字符串对象 `'hello'`，然后将现有的变量 a 来引用该字符串对象，此时字符串对象的引用计数器 `ob_refcntc` + 1
3. 在内存中创建一个列表对象，然后将现有的变量 a 来引用该列表对象，此时字符串对象的引用计数器 `ob_refcntc - 1 == 0`，会在内存中被销毁，列表对象的引用计数器  `ob_refcntc + 1`

### 常量

Python 中通常用大写来表示常量，但实际上这个值仍然是可以修改的，比如修改下面的 pi 常量

```python
import math
math.pi = 5
```

因为种种原因，Python 并没有 C++ 的 const 修饰符来保证常量的不可修改。只能通过一些特殊的操作来确保这件事

### None

空值是 Python 里一个特殊的值，用`None`表示。`None`不能理解为`0`，因为`0`是有意义的，而`None`是一个特殊的空值

## *深浅拷贝问题*

### 深浅拷贝

[python基础： 深入理解 python 中的赋值、引用、拷贝、作用域 | DRA&PHO (draapho.github.io)](https://draapho.github.io/2016/11/21/1618-python-variable/)

深浅拷贝的定义和 C++ 是一样的，**Pyhon中深浅拷贝问题是针对组合对象的**，组合对象就是这个对象中还包含其他对象，即**可嵌套其他类型的容器**，比如 list，set，dict 等，其实这是和 C++ 中的 vector、string 等容器是相同的问题

* 浅拷贝：**创建一个新的组合变量**，但是组合变量中每一个元素指向拷贝的对象内元素地址
* 深拷贝：**创建一个新的组合变量**，原对象中的每个元素都会在新对象中重新创建一次

比如说下面这种情况，完成后 a 的值会变成 `[8, [1, 9], 3]`，b 的值会变成 `[0, [1, 9], 3]` 

```python
a = [0, [1, 2], 3]
b = a[:]
a[0] = 8
a[1][1] = 9
```

<img src="组合对象的浅拷贝问题.jpg">

实现深浅拷贝的方法如下

* 组合对象 list，set，dict（没有 tuple）都自带了 copy 浅拷贝方法，而深拷贝则需要导入copy 模块，调用`deepcopy()`，copy 的 `copy` 方法对应浅拷贝

  ```python
  import copy
  a = [0, [1, 2], 3]
  b = copy.deepcopy(a)
  ```

  <img src="deepcopy深拷贝.jpg">

* 还有一些内置方法也会发生浅拷贝

  * 列表 list 的切片操作

    ```python
    original_list = [1, 2, 3]
    shallow_copy_list = original_list[:]
    ```

  * 列表 list、字典 dict 和集合 set 的构造函数

    ```python
    original_list = [1, 2, 3]
    shallow_copy_list = list(original_list)
    ```

### 可变对象 & 不可变对象对引用赋值的影响

对[不可变类型](#Python 数据类型分类)的变量重新赋值，实际上是重新创建一个不可变类型的对象，并将原来的变量重新指向新创建的对象（如果没有其他变量引用原有对象的话（即引用计数为 0），原有对象就会被回收）

因此函数值传参时如果参数是不可变类型的话效果就是传值传参，而可变类型就是传引用传参了

对象赋值 `b = a` 

* 如果 a 指向的是可更改对象，之后只要通过 a 改动其所指向的对象中的元素（因为赋值就是引用），b 所指的对象中的元素就会随之改变；反之（通过 b 进行改动）亦然

  ```python
  >>> list1 = [1, 2, 3, 4]
  >>> b = a = list1
  >>> a[2] = 8
  >>> list1
  [1, 2, 8, 4]
  >>> b
  [1, 2, 8, 4]
  ```

* 如果 a 指向的是不可更改对象，那么 a 的改动会先创建新的对象，然后只让 a 自己引用新的对象；而 b 仍然引用的是原来 a 和 b 一块引用的对象（直接从赋值的角度想就可以了）

  ```python
  >>> a = 2
  >>> print(a)
  2
  >>> b = a
  >>> a = 5
  >>> print(a)
  5
  >>> print(b)
  2
  ```

# Python虚拟机

Python是一门解释型语言，它没有编译器后端生成二进制码，需要借助虚拟机运行

# 基本语法

参考 [(19条消息) 语法：Python与C++对比_yuyuelongfly的博客-CSDN博客_python c++](https://blog.csdn.net/Cxiazaiyu/article/details/108937936)

## *命名规范*

### 标识符 Indentifier

* 第一个字符必须是字母表中的字母或下划线 `_`，标识符的其他部分由字母、数字和下划线组成
* 变量名全部小写，常量名全部大写
* 类名用大写驼峰
* 模块和包的名字用小写
* Python3支持unicode之后标识符可以用中文的，但是实际中不要这么做

### Python下划线命名规范

和 C/C++ 中 `_` 只是作为一种非强制的明明规则而已，**Python 中的 `_` 是会实际被解释器解释为特定的语法意义的**

* 单前导下划线 `_var`：作为类的私有属性，但不会被强制执行，一般只是作为一种提醒。唯一的影响是 `import` 的时候不会被导入
* 双前导下划线 `__var`：强制作为类的私有属性，无法在类外被访问，因为通过名称修饰规则修改过了，解释器对该属性的访问会发生变化
* 单末尾下划线 `var_`：按约定使用以防止与Python关键字冲突
* 双前导和双末尾下划线 `__var__`：表示Python中类定义的特殊方法，称为 dunder method 或者 magic method 魔法函数
* 单下划线 `_`：临时或者无意义的变量，可以用来接受不需要的返回值

## *运算符*

由于Python动态语言的灵活性，Python的运算符和C++中有很多细节上的区别

赋值运算符、位运算符则和C++中的完全一样

### 除法运算符

```cpp
//C
int a = 10;
int b = 3;
printf(a / b); // 3 整数除法截断
printf(a % b); // 1 取模
```

上面是C中除法和取模操作，Python的算术运算符与C的不同主要是在除法上

1. **Python 2.x 中的除法**：整数除法会直接截断小数部分，即使两个操作数都是整数，结果也会是整数，小数部分会被截断掉。这意味着 `5 / 2` 的结果是 `2`，而不是 `2.5`

2. **Python 3.x 中的除法**：除法操作符 `/` 的行为更为直观和符合数学定义。无论操作数的类型是什么，除法操作符 `/` 总是执行浮点数除法。这意味着 `5 / 2` 的结果是 `2.5`

   ```python
   In [1]: 5 / 3
   Out[1]: 1.6666666666666667
   ```

	Python提供了一个地板除 floor division `//` 来取整
	
	```python
	In [2]: 5 // 3
	Out[2]: 1
	```

`divmod()` 同时得到商和余数

```python
#Python
>>> 10 / 3
3.33333333333
>>> 10 // 3
3
>>> 10 % 3
1
>>> divmod(10, 3)
(3, 1)
```

### 其他运算符

* 算术运算符
  * `**` 求幂
  * Python 中没有单独的自增 `++` 和自减 `--` 运算符，而是使用 `+= 1` 和 `-=` 1 来实现

* 比较运算符：和C++不一样的是，Python支持连续比较，即 `a>b>c` 是可以的，而C++中只能写成 `a>b && b>c`

* 逻辑运算符：Python中没有 `$$ || !` 的逻辑运算符，而是用 `and or not` 来表示，短路原理也适用

* 成员运算符：`in` 和 `not in` 是Python独有的用来判断对象是否属于某个集合的运算符

* 身份运算符：`is` 和 `is not` 用来判断两个标识符是否引用同一个对象，类似于 `id()`。`==` 只是用来判断值相等

* 三目运算符：类似于 C++ 的 `cond ? a : b`为真时的结果 if 判定条件 else 为假时的结果

  ```python
  a if condition else b
  ```
  
  这里一定要写 else，不能只写 if

## *终端IO*

### 字符串

* Python中的字符串用单引号或者双引号没有区别，但是用json格式转换的时候还是要考虑这个问题
* 原生字符串：在字符串前加r或R取消转义字符的作用，如 `\n` 会被取消换行的意思
* unicode字符串和byte类型字符串：在字符串前分别加u或U和b
* 字符串会自动串联，即 `'I' "love" "you"` 会被自动转换为 `"I love you"`

### 内置函数 `input()` 输入

* 读到的是一个字符串
* 去除空白的开头 `lstrip()`，去除结尾的空白用 `rstrip()`，去除两端的空白用 `strip()`

### `print()` 输出函数

内置函数 `print()` 打印到屏幕。注意：Python3中的括号必不可少，因为它实质上是call函数，而Python2中则可以不用括号

### `print()` 的格式控制

* C语言式
* `str.format()`

## *控制流*

Python中给没有 switch - case

### 条件判断

```python
if condition_1:
    statement_block_1
elif condition_2:
    statement_block_2
else:
    statement_block_3
```

注意：Python 中是灭有 else if 的写法的，必须使用 elif

### 循环

* for循环
  * for循环通常用来遍历可迭代的对象，如一个列表或者一个字典，语法 `for <variable> in <sequence>:`
  
  * 使用range自动生成，见下
  
  * 使用内置的 `enumerate()` 来进行循环计数或者C++/Java风格的迭代索引：`for count, value in enumerate(values):`
  
  * Python循环中可以同时引用多个变量
  
    ```python
    for i, value in enumerate(['A', 'B', 'C']):
        print(i, value)
    ```
  
* while循环：和C++不同，while可以增加else

* 循环控制：用break跳出当前层的循环，continue跳过本次循环的剩余部分，return跳出所有循环

### range

`range(start, stop[, step])`

* 自动生成一个序列区间给for，生成的序列区间是坐闭右开的，即 `range(0,100)` 生成的是0-99，间隔为1的序列
* 可以指定步长 `range(0, 2, 100)`

注意点

- `range()` 返回的是一个 "range object"，而不是实际的列表类型。使用 `list()` 函数可以将其转换为列表。
- 在 Python 3.x 中，`range()` 生成的是一个惰性序列，意味着它会在你遍历它时才生成每一个数字，这在处理大范围的数字时更加高效。
- `range()` 不支持浮点数做为参数，如果需要步长为浮点数的范围，可以考虑使用 `numpy` 库的 `arange()` 函数或者自定义一个生成器

# 数据类型

## *Intro*

### Python 数据类型分类

* 根据内置和自定义分
  * 内置：数字、字符串、布尔、列表、元组、字典、Bytes、集合
  * 自定义：类
* 根据可变类型和不可变类型分
  * 可变 mutable：列表 list、字典 dict、集合 set
  * 不可变 immutable / rebinding：数字 int, long、字符串 str、布尔 bool、元组 tuple

### 打印数据类型

内建的 `type()` 函数来查看任何变量的数据类型

## *数字类型*

### 整数 int

Python 的整数长度为 32 位，相当于 C 中的 long，并且通常是连续分配内存空间

* 对象池/缓存的作用：Python 会**缓存**使用非常频繁的小整数-5至256、ISO/IEC 8859-1单字符、只包含大小写英文字母的字符串，以对其复用，不会创建新的对象

  可以通过下面的例子来验证，可以看到 -4 和 -5 的内存地址相差 32 位，而 -6 的地址就离的比较远了

  ```python
  >>> id(-4)
  140405897304208
  >>> id(-5)
  140405897304176
  >>> id(-6)
  140405901250416
  ```

* 整数缓冲区：刚刚被删除的整数的内存不会被立刻回收，而是在后台缓冲一段时间等待下一次的可能调用

整数类型还有浮点数 float和复数

### 类型转换

和 C++ 一样，Python 也有隐式和显示（强制）类型转换，用法也差不多

### Bool值

* True 和 False

* 一个常范的错误是None不是bool值，二是一个单例类对象

  * None是逻辑运算的重要组成部分，它的类型是 NoneType。不能对None进行数值操作

  * None是一个单例类 Singleton，而在pep8中规定了对单例类的比较要用对象比较符 `is` 和 `is not`，而不是用数值运算符 `==`

    >"Comparisons to singletons like None should always be done with 'is' or 'is not', never the equality operators."

### 极大极小值

在Python中，可以使用浮点数的 `float` 类型来表示无穷大。可以使用 `float('inf')` 来表示正无穷大，使用 `float('-inf')` 来表示负无穷大

```python
positive_infinity = float('inf')
negative_infinity = float('-inf')

print(positive_infinity > 1000)  # 输出: True
print(negative_infinity < -1000) # 输出: True

# 无穷大值的一些性质:
print(positive_infinity + 1)     # 还是无穷大
print(-positive_infinity)         # 变成负无穷大
print(positive_infinity * 2)      # 还是无穷大
print(positive_infinity / 2)      # 还是无穷大
```

注意：处理无穷大时要小心，因为它会根据算术运算规则参与计算，并可能导致一些不直观的结果（比如 `inf - inf` 结果是 `nan`，即非数字（not a number））

```python
>>> a = float('inf')
>>> b = float('inf')
>>> a - b
nan
```


## *列表 List*

### 与 `std::vector` 的区别

Python的列表是一个可变长度的顺序存储结构

列表类似C++的 `std::vector`，区别在于列表的每一个位置存放的都是对象的引用，同时因为动态特性不需要指定存储元素的类型，所以它可以混数据类型存储

### 列表操作、内置方法

注意python的list也是左闭右开的，比如 `[0:-1]` 是从第一个取到倒数第二个元素

https://www.liujiangblog.com/course/python/19

<img src="list方法.png" width="80%">

* 构造

  * `a=[]` 或者 `a=list()`空列表
  * 从构造器
  * 列表推导式

* 插入

  * append
  * insert

* 删除

  * `list.remove(obj)` 用于移除列表中某个值的第一个匹配项

    ```python
    aList = [123, 'xyz', 'zara', 'abc', 'xyz'];
    aList.remove('xyz');
    ```

  * `list.pop([index=-1])` 移除列表中的一个元素（默认最后一个元素），并且返回该元素的值

* 排序：`list.sort( key=None, reverse=False)`

  * key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
  * reverse -- 排序规则，**reverse = True** 降序， **reverse = False** 升序（默认）

## *元组 Tuple*

用方括号 `[]` 扩起来的是列表，用圆括号 `()` 括起来的元组。元组就是不可变的列表

### 元组中允许的操作

* 使用方括号加下标访问元素
* 切片（形成新元组对象）
* `count()`/`index()`
* `len()`/`max()`/`min()`/`tuple()`

### 元组中不允许的操作

* 修改、新增元素
* 删除某个元素（但可以删除整个元组）
* 所有会对元组内部元素发生修改动作的方法。例如，元组没有 remove，append，pop 等方法

### 元组解包 & `*` 操作符

元组解包 Tuple Unpacking 是指将一个包含多个元素的元组分解成多个变量的过程

Python 中 `*` 除了 [动态参数](#动态参数) 中的作用外，另外一个作用就是在元组解包。元组解包有三种形式

1. 解包可迭代对象（如列表、元组等），将其元素作为单独的参数传递给函数

   ```python
   numbers = [1, 2, 3]
   print(*numbers)  # 相当于 print(1, 2, 3)
   
   subtree = nx.DiGraph()
   for edge in nx.edge_bfs(G, root): # returns a generator
   	subtree.add_edge(*edge) # tuple unpacking
   ```

2. 赋值给多个变量的元组解包。注意：这种情况不要在元组前加 `*`，会自动拆包的

   ```python
   >>> my_tuple = (1, 2, 3, 4, 5)
   >>> a, b, c, d, e = *my_tuple
     File "<stdin>", line 1
   SyntaxError: can't use starred expression here
   >>> a, b, c, d, e = my_tuple
   >>> print(a)
   1
   ```

3. **扩展的迭代解包**：从 Python 3.5 开始，`*`也可以用于扩展的迭代解包（Extended Iterable Unpacking），其中可以在赋值操作时使用`*`来接收一个序列的多余元素

   ```python
   first, *middle, last = [1, 2, 3, 4, 5]
   print(first)  # 输出 1
   print(middle) # 输出 [2, 3, 4]
   print(last)   # 输出 5
   ```

### 元组的坑

元组只保证它的一级子元素不可变，对于嵌套的元素内部，不保证不可变

```python
>>> tup = (1, 2, [3, 4])
>>> tup[2][1] = 8
>>> tup
(1, 2, [3, 8])
```

## *字符串 String & bytes*

**Python3在运行时全部使用Unicode编码**，所以不会有Python2中的编码问题，可以放心使用中文

### 字符串

**字符串是不可变类型**，但支持对它的切片和取子串（相当于就是产生了一个新对象）

字符串可以用单引号 `''` 或双引号 `""` 括起来表示。Python 3 还支持三重引号 `''' '''` 或 `""" """`，用于跨多行的字符串表示。**没有固定的规则要求使用单引号还是双引号**，一条默认的规则是如果字符串中包含了单引号转义字符，最好使用双引号；反之如果字符串中包含了双引号转义字符，则最好使用单引号

```python
string1 = 'Hello, World!'
string2 = "Python Programming"
string3 = '''This is a
multi-line string.'''
```

### 字符串函数

Python 的字符串操作十分强大

注意：因为 Python 的字符串是不可变对象，所以所有的字符串方法都不会直接修改原始字符串，而是返回一个新的字符串。如果想要使用修改后的字符串，就必须将返回的新字符串赋值给一个变量（无论是同一个变量还是一个新变量）

字符串的常用基本操作有下面这些，更完整的可以看 https://www.liujiangblog.com/course/python/21

* 用加号运算符 `+` 将两个字符串连接起来

* 用内置函数`len()`获取字符串的长度

* 可以通过索引访问字符串中的单个字符，也可以通过切片获取子字符串

  ```python
  my_string = "Python Programming"
  print(my_string[0])      # 输出：P
  print(my_string[7:18])   # 输出：Programming
  ```

* `upper()`：将字符串中的所有字母转换为大写

* `lower()`：将字符串中的所有字母转换为小写

* `strip()`：去除字符串两端的空白字符，默认情况下，这些空白字符包括：

  - 空格 `' '`
  - 制表符 `'\t'`
  - 换行符 `'\n'`
  - 回车符 `'\r'`
  - 垂直制表符 `'\v'`
  - 换页符 `'\f'`

  ```python
  s = '   hello world \n'
  clean_s = s.strip()
  print(clean_s)  # 输出: 'hello world'
  ```

  `strip()` 也可以接受一个参数（字符串），该参数指定了要从字符串两端移除的字符集合。当提供这个参数时，函数会移除字符串两端所有出现在参数中的字符，直到遇到一个不在参数中的字符为止

  ```python
  s = '.#.#..hello world#..#.'
  clean_s = s.strip('.')
  print(clean_s)  # 输出: '#.#..hello world#..#'
  ```

* `split()`：将字符串拆分为子字符串列表，这个方法会根据指定的分隔符对原始字符串进行切割，并返回一个包含这些子字符串的列表

  ```python
  str.split(sep=None, maxsplit=-1)
  ```

  - `sep`：指定用作分隔符的字符串。如果未提供或指定为 `None`，则按空白字符（空格、换行 `\n`、制表符 `\t` 等）进行分割
  - `maxsplit`：可选参数，指定分割的次数。默认情况下（即当其值为 `-1` 时），分割动作会执行尽可能多的次数。如果指定了 `maxsplit`，则只会分割为 `maxsplit+1` 个子字符串

  ```python
  >>> s = "apple,banana,cherry"
  >>> print(s.split(','))
  ['apple', 'banana', 'cherry']
  >>> print(s.split())
  ['apple,banana,cherry' # 没有成功split
  ```

* `join()`：将序列中的元素连接为一个字符串

  ```python
  separator.join(iterable)
  ```

  - `separator` 是希望在每个元素之间插入的字符串（可以为空）
  - `iterable` 是包含要连接的字符串的可迭代对象

  ```python
  >>> ''.join(['hello','world'])
  'helloworld'
  >>> ' '.join(['hello','world'])
  'hello world'
  >>> '-'.join(['hello','world'])
  'hello-world'
  ```

* `replace(old_str, new_str)`：替换字符串中的指定子字符串

* `find()`：查找字符串中是否包含指定子字符串，返回子字符串的第一个索引值

* `count()`：计算指定子字符串在字符串中出现的次数

### 删除字符串中的某个char



### 逆序字符串

Python没有builtin的逆序字符串方法

* reversed返回一个iterator

### 字符串格式化

使用字符串的`format()`方法或f-strings进行字符串格式化

```python
name = "Alice"
age = 30
print("My name is {} and I am {} years old.".format(name, age))
# 输出：My name is Alice and I am 30 years old.

print(f"My name is {name} and I am {age} years old.")
# 输出：My name is Alice and I am 30 years old.
```

### bytes

在Python3以后，字符串和bytes类型彻底分开了

字符串是以字符为单位进行处理的，bytes类型是以字节为单位处理的

### f-string

f-string，即格式化字符串 formatted string，从 Python 3.6 开始引入的 f-string 提供了一种更简洁易读的方式来嵌入 Python 表达式到字符串常量中

当在字符串前加上 `f` 或 `F` 前缀，并将变量或表达式放在大括号 `{}` 中时，Python 会在运行时计算这些表达式的值，并将它们插入到字符串中。因此，f-string 是构建动态字符串时的便捷方法

```python
proj_dir = "/path/to/project"
proj_name = "MyProject"
case_list_file = "cases.txt"

# 使用 f-string 将变量插入到字符串中
info_str = f'mff_dir: {proj_dir}, proj_name: {proj_name}, case_list_file: {case_list_file}'

print(info_str)
```

输出将会是

```
mff_dir: /path/to/mff, proj_name: MyProject, case_list_file: cases.txt
```

注意：f-string中如果要打印 `{}` 本身的话得套一层 `{}` 来转义

## *字典 & 集合*

### 字典和 `std::unordered_map` 的区别

```python
d = { 'apple' : 1, 'pear' : 2, 'peach' : 3 }
```

Python的字典数据类型是和 `std::unordered_map` 一样是基于hash算法实现的

**从Python3.6开始，字典变成有序的，它将保持元素插入时的先后顺序**

### 字典

* 字典核心API

  <img src="字典方法.png" width="80%">

  注意：`keys()` 返回的是一个不支持索引的 dict_keys，可以用 `list()` 转换为 list 使用

* 字典键的访问

  * dict：当尝试对字典中一个不存在的键**赋值**的时候，不会报 KeyError，而是直接创建这个 pair，但是若尝试**获取或者修改**一个不存在于普通字典中的键，Python 将会抛出 KeyError。避免 KeyError 的方法是
  
    * 使用 `get()` 获取字典键的值，该方法允许返回一个默认值而不是抛出 KeyError
  
      ```python
      value = normal_dict.get('a', 0)  # 如果 'a' 不存在，返回0
      ```
  
    * 在尝试读取或更新之前手动检查键是否存在
  
      ```python
      if 'a' in normal_dict:
          normal_dict['a'] += 1
      else:
          normal_dict['a'] = 1
      ```
  
    * 使用 `setdefault` 方法，它会设置并返回指定键的值，如果键不存在，则首先设置为默认值
  
      ```python
      normal_dict.setdefault('a', 0)
      normal_dict['a'] += 1
      ```
  
    * 捕获 `KeyError` 异常
  
      ```python
      try:
          normal_dict['a'] += 1
      except KeyError:
          normal_dict['a'] = 1
      ```
  
    ```python
    dic = {'Name': 'Jack', 'Age': 7, 'Class': 'First'}
    
    # 1  直接遍历字典获取键，根据键取值
    for key in dic:
        print(key, dic[key])
    
    # 2  利用items方法获取键值，速度很慢，少用！
    for key,value in dic.items():
        print(key,value)
    
    #3  利用keys方法获取键
    for key in dic.keys():
        print(key, dic[key])
    
    #4  利用values方法获取值，但无法获取对应的键
    for value in dic.values():
        print(value)
    ```
  
  * defaultdict 的行为类似于 C++ 的 unordered_map，即其主要特点是提供了一个默认值，用于在尝试访问字典中不存在的键时自动生成默认条目。这是它与常规字典 `dict` 的主要区别。当使用一个非嵌套的 defaultdict 时，它的行为是：
  
    * 当访问一个不存在的键时，`defaultdict` 会自动为该键创建条目，并将其值设置为通过调用在初始化 `defaultdict` 对象时提供的默认工厂函数得到的值。
    * 默认工厂函数在 `defaultdict` 被创建时作为第一个参数给出，例如：`defaultdict(list)` 将会对每个不存在的键创建一个新的空列表作为默认值字典的遍历
  
    ```python
    from collections import defaultdict
    
    # 使用普通的 dict
    d = {}
    print(d.get('key', 'default'))  # 输出: default
    # d['key']  # 这会抛出 KeyError，因为 'key' 不存在
    
    # 使用 defaultdict
    dd = defaultdict(lambda: 'default')
    print(dd['key'])  # 输出: default
    # 因为 'key' 不存在，defaultdict 调用提供的 lambda 函数生成默认值 'default'
    
    # 再次访问同一个键，可以看到它已经被设置了默认值
    print(dd['key'])  # 输出: default
    ```

### 集合

https://www.liujiangblog.com/course/python/24

集合数据类型的核心在于**自动去重**。Python的集合使用大括号 `({})` 框定元素，并以逗号 `,` 进行分隔。注意⚠️：如果要创建一个空集合，必须用 `set()` 而不是 `{}`，因为后者创建的是一个空字典

```python
>>> s = set({1, 2, 2, 3, 3, 4})
>>> s
{1, 2, 3, 4} # 去重了
>>> s = set("hello")
>>> s # 对于字符串，集合会把它一个一个拆开，然后去重
{'e', 'l', 'h', 'o'}
```

* `add(key)` or `update(key)`
* `remove(key)`
* `pop()`

## *推导式*

推导式 omprehension 是 Python 语言特有的一种语法糖，可以写出较为精简的代码

### 列表推导式

在 C++ 中生成一个存储 1-9 的平方 vector 的思路是

```cpp
vector<int> v = {}
for (int i = 1; i < 10; i++) {
	v.push_back(i*i)
}
```

但是在 Python 中生成这样的一个 list 可以用一行代码完成，即

```python
>>>lis = [x * x for x in range(1, 10)]
[1, 4, 9, 16, 25, 36, 49, 64, 81]
```

推导式还可以有很多其他的花样

* 增加条件语句

  ```python
  >>>lis = [x * x for x in range(1, 10) if x % 2==0]
  [4, 16, 36, 64]
  ```

* 多重循环

  ```python
  >>> [a + b for a in '123' for b in 'abc']
  ['1a', '1b', '1c', '2a', '2b', '2c', '3a', '3b', '3c']
  ```

  最前面的循环就是最内层的循环

```python
matched = [False] * len(s)
```

* `[False]`: 创建一个只包含单个元素 `False` 的列表
* `[False] * len(s)`: 将包含 `False` 的列表重复 `len(s)` 次，生成一个新的列表，这个新列表的每个元素都是 `False`，并且其长度为输入字符串 `s` 的长度

### 字典与集合推导式

```python
dic = {x: x**2 for x in (2, 4, 6)}
s = {x for x in 'abracadabra' if x not in 'abc'}
```

### 元组推导式

所谓的元组推导式就是生成器 generator 对象，具体内容可以查看生成器部分

```python
>>> g = (x * x for x in range(1, 10))
>>> g
<generator object <genexpr> at 0x7fbaea0dcac0>
```

# 数据结构的实现

## *链表*

## *栈 & 队列*

Python没有实现独立的栈和队列数据结构，而是利用列表的 append、pop 等操作间接实现的

### 栈

### 队列

Python中可以使用 `collections.deque` 来高效地实现队列

### 优先级队列

## *树*

## *图*

[18_图与图的遍历 - Python 数据结构与算法视频教程 (python-data-structures-and-algorithms.readthedocs.io)](https://python-data-structures-and-algorithms.readthedocs.io/zh/latest/18_图与图的遍历/graph/)

[Python中的顺序表 | 数据结构与算法（Python） (jackkuo666.github.io)](https://jackkuo666.github.io/Data_Structure_with_Python_book/chapter2/section4.html)

[数据结构必会｜图的基本概念及实现（Python）-阿里云开发者社区 (aliyun.com)](https://developer.aliyun.com/article/1155600)

## *Networkx图库*

[NetworkX — NetworkX documentation](https://networkx.org/)

[PythonGraphLibraries - Python Wiki](https://wiki.python.org/moin/PythonGraphLibraries)

Python社区提供了一些实现了图结构的库，比如Networkx、graphlib等，下面会介绍Networkx的使用

```cmd
pip3 install networkx
```

```python
import networkx as nx
```

### 建图

* 创建图数据结构

  ```python
  G = nx.Graph() # 无向图
  DG = nx.DiGraph()
  ```

* 增加节点

  ```python
  G.add_node(1)            # single node
  G.add_nodes_from([2, 3]) # nodes from iterable container
  G.add_nodes_from([(4, {"color": "red"}), (5, {"color": "green"})]) # nodes with attributes
  ```

* 增加边

  当使用`add_edge()` 添加一条边时，如果指定的任何节点（无论是起点还是终点）此前并不存在于图`G`中，networkx 会自动创建这些节点

  ```python
  Graph.add_edge(u_of_edge, v_of_edge, **attr)
  ```

  The following all add the edge e=(1, 2) to graph G:

  ```
  >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
  >>> e = (1, 2)
  >>> G.add_edge(1, 2)  # explicit two-node form
  >>> G.add_edge(*e)  # single edge as tuple of two nodes
  >>> G.add_edges_from([(1, 2)])  # add edges from iterable container
  ```

  Associate data to edges using keywords:

  ```
  >>> G.add_edge(1, 2, weight=3)
  >>> G.add_edge(1, 3, weight=7, capacity=15, length=342.7)
  ```

* 权重

* 从图中删除元素

### 查看图的性质

* 整图的性质
  * `info()` ：获取图的基本信息，包括节点数、边数等
  * `nx.generate_adjlist(G)`：可以打印出图的邻接表，显示节点之间的连接关系
* 节点的性质
  * `in_degree()` 和 `out_degree()`：获取某个节点（返回一个int）或整图（返回一个含有 `(节点序号, 度)` tuple 的迭代器）的入度/出度
  * `degree()` 用于获取无向图的度，同样可以获取一个节点或整图节点的度
* 边的性质

### 图算法

topological_sort

### 绘图

networkx的绘图功能是利用matplotlib来实现的

* 绘图方程：`draw()` 提供了比较简单的绘图，`draw_networkx()` 则提供了更多的特性

* 图布局算法

  <img src="graph_layout.png" width="70%">

# 函数

Python函数最大的特点是可以在函数中再定义函数，因为Python一切皆对象，函数也是一个类，所以是相当于定义了一个内部类

## *函数基础*

### 定义

```python
def 函数名(参数):
    # 内部代码
    return 表达式 
```

### 传引用传参 Pass by reference

Python变量是对象的引用，因此函数传参也采取的是传引用传参

```python
a = 1 # 不可变
b = [1, 2, 3, 4] # 可变
def func(a, b):
    print("在函数内部修改之前,变量a的内存地址为： %s" % id(a))
    print("在函数内部修改之前,变量b的内存地址为： %s" % id(b))
    a = 2
    b[3] = 8
    print("在函数内部修改之后,变量a的内存地址为： %s" % id(a))
    print("在函数内部修改之后,变量b的内存地址为： %s" % id(b))
    print("函数内部的a为： %s" % a)
    print("函数内部的b为： %s" % b)


print("调用函数之前,变量a的内存地址为： %s" % id(a))
print("调用函数之前,变量b的内存地址为： %s" % id(b))
func(a, b)
print("函数外部的a为：%s" % a)
print("函数外部的b为：%s" % b)
```

<img src="传引用传参试验.png">

* 不论是可变量a还是不可变量b，调用函数和调用函数后但未修改的内存地址一样，证实了传引用传参
* 函数内部修改后，根据之前变量与引用的关系，a被重新指向了2这个数值对象，而b则没有改变引用对象
* 试验里这种将实参和形参命名相同是一种不好的代码习惯，最好使用不同的名称避免混淆

### 返回

因为Python没有变量类型，所以也就不需要写return的类型了，默认返回为None

和C语言函数只能返回一个值不同，Python函数可以同时返回多个值并且同时接收

```python
def func(x, y, z):
  	# ...
    return x, y, z
a, b, c = func(x, y, z)
```

Python3.6 中引入了类型提示 type hints `:` 并且 Python3 引入了 `->` 来声明返回类型

```python
# Python2
class Solution(object):
    def search(self, nums, target):

# Python3
class Solution:
    def search(self, nums: List[int], target: int) -> int
```

## *参数*

### 必传参数

必传参数又称位置参数或顺序参数，是必须在call函数的时候按照顺序提供的参数

但是若在传参的时候指定了参数名的时候，顺序也是可以调换的

```python
def student(name, age, classroom, tel, address="..."):
    pass

student(classroom=101, name="Jack", tel=66666666, age=20)
```

### 默认参数 Default parameter

* 默认参数必须要在必传参数之后
* 当有多个默认参数的时候，通常将更常用的放在前面，变化较少的放后面
* 在调用函数的时候，尽量给实际参数提供默认参数名，否则可能会混淆犯错

### 动态参数

动态参数，是指不限定传入参数个数的参数包，**必须放在所有的位置参数和默认参数后面**

* `*args`：会将实际参数打包成一个元组传入形式参数。如果参数是个列表，会将整个列表当做一个参数传入
* `**kwargs` (keyword arguments)：两个星表示接受键值对的动态参数（没有被命名的KV参数），数量任意。调用的时候会将实际参数打包成字典

```Python
def print_all_args(*args, **kwargs):
    for arg in args:
        print(arg)
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_all_args('pos1', 'pos2', arg1='value1', arg2='value2')
# 输出：
# pos1
# pos2
# arg1: value1
# arg2: value2
```

### 万能参数

### 类型建议符 Python 3.5

```python
def sum(num1: int, num2: int) -> int:
    return num1 + num2
```

函数参数中的冒号是参数的类型建议符，告诉程序员希望传入的实参的类型。函数参数列表后的箭头是函数返回值的类型建议符，用来说明该函数返回的值是什么类型

类型建议符并非强制规定和检查，即使传入的实际参数与建议参数不符，也不会报错。类型建议符号的主要作用是令使用者快速了解函数的作用，没有对类型的解释说明的话，可能会需要花费更多的时间才能看出函数的参数和返回值是什么类型，有了说明符，可以方便程序员理解函数的输入与输出

## *变量作用域*

### 作用域分类

作用域就是一个 Python 程序可以访问到的命名空间的正文区域

```python
x = int(2.9)  # 内建作用域，查找 int 函数

global_var = 0  # 全局作用域
def outer():
    out_var = 1  # 闭包函数外的函数中
    def inner():
        inner_var = 2  # 局部作用域
```

LEGB 规则：Python以 L-E-G-B 的顺序依次寻找变量

* 局部作用域 Local scope：定义在函数内部的变量拥有局部作用域，局部变量只能在其被声明的函数内部访问
* 闭包作用域 Enclosing scope
* 全局作用域 Global scope：定义在函数外部的变量拥有局部作用域
* 内置作用域 Built-in scope： 包含了内建的变量/关键字等，最后才会被搜索

### 代码块的作用域

在 Python 中，for 循环中定义的变量在循环结束后依然可以访问，因为 Python 中的作用域规则与 C++ 不同。这意味着即使变量是在 for 循环内部被首次赋值的，循环完成后仍然可以在外部使用这些变量

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

# 初始化最大和最小值变量
max_value = None
min_value = None

# 遍历数字列表并更新最大和最小值变量
for number in numbers:
    if max_value is None or number > max_value:
        max_value = number
    if min_value is None or number < min_value:
        min_value = number

# for循环结束后，仍可以访问min_value和max_value变量，并打印结果
print(f"The maximum value in the list is: {max_value}")
print(f"The minimum value in the list is: {min_value}")
```

## *Python 的全局 & 局部变量陷阱*

### 赋值导致局部化

首先要明确一个概念**赋值导致局部化**：函数内的变量是否是局部变量的决定是在编译时（或者说定义时）做出的，而不是在运行时。当定义一个函数时，Python 解释器会扫描函数体内的所有语句来决定每个变量的作用域

如果在函数内对一个变量进行了赋值操作，即使在赋值之前已经有同名的全局变量存在，Python 也会将它视作一个新的局部变量

比如说下面这个错误的闭包

```python
def outer(arg):
    temp = 10
    def inner():
        _sum = temp + arg
        temp += 1   # 在内函数中尝试改变temp的值
        print('_sum = ', _sum)
        return _sum
    return inner
```

在执行 `_sum = temp + arg` 时，如果函数内部存在对 `temp` 的赋值语句（如 `temp += 1`），Python 解释器会认为 `temp` 是一个局部变量，即使它在之前并没有被显式定义

当 Python 函数被定义时，解释器扫描函数的代码块，并确定哪些变量是局部的。如果在函数中给一个变量赋值了，那么该变量就被认为是局部变量，除非使用了 `global` 或 `nonlocal` 关键字声明其作用域。这意味着，就算 `temp` 在 `temp += 1` 行下面，仍然会影响到上面的 `_sum = temp + arg` 行

由于函数定义时 `temp` 被视为局部变量，并且在 `_sum = temp + arg` 这一行尝试读取 `temp`，但是在这一行执行时，`temp` 还没有在局部作用域中被赋予任何值。因此，当尝试计算 `_sum = temp + arg` 时，Python 解释器抛出 `UnboundLocalError`，因为它找不到局部变量 `temp` 的值

再举一个例子

```python
var = 'global'
>>> my_func()
>>> print(var)
global
>>> def my_func():
...     print(var)
...     var = 'locally_changed'
...
>>> var = 'global'
>>> my_func()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in my_func
UnboundLocalError: cannot access local variable 'var' where it is not associated with a value
```

上面的代码中，my_func 中的 var 也是因为在第二行被赋值了，所以被视作一个局部变量了，所以在第一行访问 var 的时候就报了未定义的错误

### `global` 和 `nonlocal` 关键字

* global 可以在任何地方指定当前变量使用外部的全局变量

  ```python
  a = 10
  b = 6
  def fun():
      global a 
      a = 2 # 这里的a就是上面的a，修改了全局a的值
      b = 4 # 局部变量b，和外面的b灭有关系
  ```

  Python 中，一个函数可以未经 global 声明地任意读取全局数据，但要赋值修改时必须符合如下条件之一：

  1. 全局变量使用 global 声明
  2. 全局变量是可变类型数据

  但是也存在例外 [python基础： 深入理解 python 中的赋值、引用、拷贝、作用域 | DRA&PHO (draapho.github.io)](https://draapho.github.io/2016/11/21/1618-python-variable/)

* nonlocal 和 global 的区别在于 nonlocal 是用于嵌套作用域的，即作用范围仅对于**所在子函数的上一层函数**中拥有的局部变量，必须在上层函数中已经定义过，且非全局变量，否则报错

  一个典型的使用场景是修正上面的闭包

  ```python
  def outer(arg):
      temp = 10
      def inner():
          nonlocal temp  # 使用 nonlocal 声明 temp 来自外部作用域
          _sum = temp + arg
          temp += 1
          print('_sum = ', _sum)
          return _sum
      return inner	
  ```

### 默认参数陷阱

## *匿名函数/lambda表达式*

`lambda arguments : expression`

- `arguments` 是传递给 `lambda` 函数的参数。它们可以是单个参数，也可以是用逗号分隔的多个参数
- `expression` 是在函数调用时将被评估并返回其值的表达式

```python
add = lambda x, y: x + y
print(add(5, 3))  # 输出: 8
```

定义了一个执行加法的 lambda 函数，并存储在变量 `add` 中

## *Stub File*

上面提到了Python3.6 中引入了 type hint

Stub file 存根文件，即通过第三方文件`.pyi`文件，定义函数参数类型与返回值类型

### API说明

```python
ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
```

- `name or flags...`: 这里的省略号（`...`）意味着可以传递一个或多个参数名称（例如 `'foo'` 或 `'--foo'`）。它们可以是位置参数的名称或是可选参数的标志
- `[argument]`: 方括号通常用于表示参数是可选的，即在调用这个方法时，可以选择是否提供这个参数。如果方括号内部有一系列参数，比如 `[, action][, nargs]`，这表示 `action` 和 `nargs` 都是可选参数
- `[, action]`: 你可以包含 `action` 参数，也可以不包含。如果选择包含此参数，那么应该为它提供一个值（例如 `store`, `store_true` 等）。如果不包含它，将使用默认行为

## *built_in functions*

[Built-in Functions — Python 3.12.4 documentation](https://docs.python.org/3/library/functions.html)

<img src="Python_builtin_functions.png">

# 三器一闭

## *迭代器 Iterator*

和C++一样，Python中一般的数据结构 list/tuple/string/dict/set/bytes 都是可以迭代的数据类型，也可以为自定义类对象实现迭代器

### 实现迭代器协议的方法

1. **`__iter__(self)`**
   - 返回迭代器对象本身
   - 通常用于 `for` 循环和其他需要迭代的场景中
2. **`__next__(self)`**
   - 返回容器的下一个元素
   - 当没有更多元素时，应抛出 `StopIteration` 异常

下面给出实现一个自定义迭代器的简单例子

* Python 中的 for 循环和 C++11 的范围 for 一样，就是**直接调用可迭代对象的 `__iter__()` 得到一个迭代器对象，然后不断地调 `next()`**
* 通过 collections 模块的 `iterable()` 函数来判断一个对象是否可以迭代

```C++
class Counter:
    def __init__(self, low, high):
        self.current = low
        self.high = high

    def __iter__(self):
        return self

    def __next__(self):
        if self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return self.current - 1

# 创建迭代器
counter = Counter(1, 3)

# 使用迭代器
for c in counter:
    print(c)

```

### 通过内置函数来操作迭代器

Python 还提供了一些内置函数3来生成和操作迭代器

- `iter(object[, sentinel])`：获取一个可迭代对象的迭代器

- `next(iterator[, default])`：调用迭代器的 `__next__()` 方法获取下一个元素

- `range(start, stop[, step])`：返回一个范围内连续整数的迭代器

  注意：range 是一个可迭代对象，而不是一个迭代器，可以看 [只实现 iter，不实现 next 有意义吗？](#只实现 iter，不实现 next 有意义吗？)

下面是一个对内置对象使用迭代器方法的简单例子

```python
>>> list1 = [1, 2, 3]
>>> it = iter(list1)
>>> it
<list_iterator object at 0x7fe2cdc2aa30>
>>> next(it)
1
>>> next(it)
2
>>> next(it)
3
>>> next(it)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
StopIteration
```

**利用 `iter()` 创建迭代器对象，然后利用 `next()` 取下一个元素，直到没有元素跳StopIteration异常**

### 迭代器 Iterator 和可迭代 iterable 的区别

<img src="iterable_iterator.png" width="65%">

* **凡是可作用于 for 循环的对象就是可迭代类型**，因为 for 关键字会直接调用可迭代对象的 `__iter__()` 得到一个迭代器对象

  * 可以通过 `collections.abc` 模块的 `Iterable` 类型判断是否是可迭代类型

    ```python
    >>> from collections.abc import Iterable
    >>> isinstance('abc', Iterable) # str是否可迭代
    True
    >>> isinstance([1,2,3], Iterable) # list是否可迭代
    True
    >>> isinstance(123, Iterable) # 整数是否可迭代
    False
    ```

* **凡是可作用于 `next()` 函数的对象都是迭代器类型**，因为 `next()` 会调用迭代器对象的 `__next__()` 函数

* Python 的数据结构对象 list、dict、str 等都是可迭代类型，而不是迭代器，因为它们可以用于 for 关键字迭代而不能作为 next 函数的对象


用 C++ 来类比一下，迭代器类型就是 C++的 `::iterator` 类，不过 C++ 可以直接这么取，不需要通过 `iter()` 才能获得迭代器对象，同时 C++ 可以直接通过 `++, --` 控制迭代器方向，并通过解引用获取下一个 value，而 Python 的迭代器对象则需要通过 `next()` 来获取预先设计好的 “下一个” value

<img src="iterable_iterator_generator.png" width="80%">

注意：迭代器 iterator 一定是可迭代类型 iterable，因为迭代器同时实现了 iter 和 next；可迭代类型则不一定是迭代器，因为可迭代类型只需要实现 iter，它可以像 range 一样里面再包其他的可迭代类型/迭代器

### 只实现 iter，不实现 next 有意义吗？

换句话说，只是一个可迭代类型，而不是一个迭代器有意义吗？

`range` 本身并不是一个迭代器，而是一个可迭代对象 iterable

当调用 `range()` 时，Python 不会立即创建一个列表，并把所有元素载入内存，而是返回一个 range 对象。这个对象会按需计算每个元素（类似于迭代器），但它并不符合迭代器协议，因为它没有实现 `__next__()` 方法。相反，`range` 实现了 `__iter__()` 方法来返回一个它管理的元素的迭代器，通过这些迭代器实际上会在迭代过程中按需生成值

通过调用 `range()` 函数返回的对象的 `__iter__()` 方法，可以获取到一个真正意义上的迭代器。这个迭代器对象会遵循迭代器协议，实现 `__iter__()` 和 `__next__()` 方法

下面是使用 range 的一个例子

```python
r = range(5)
print(type(r))  # 输出: <class 'range'>

# 获取 range 可迭代对象
it = iter(r)
print(type(it))  # 输出: <class 'range_iterator'>

# 迭代 range 对象
for num in r:
    print(num)  # 输出: 0 1 2 3 4

# 使用迭代器直接迭代
while True:
    try:
        print(next(it))  # 输出: 0 1 2 3 4
    except StopIteration:
        break
```

也就是说，调用关系是

```
range -> range_iterator （虽然名字叫iterator，但是是一个 iterable）-> 容器内部真正的迭代器 iterator 
```

我们可以实现一个自己的 range

```python
class MyRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

# 使用 MyRange 类
for number in MyRange(1, 4):
    print(number)
```

### 迭代器 & 生成器的优势：懒加载

一个很自然的问题是，干嘛要用？在笔者看来，有以下三个好处

1. 和 C++ 一样的作用，同时创建多个迭代器 / 生成器对象，保存不同的遍历状态，方便遍历

2. 按需加载（懒加载）节省内存，这个可能是最重要的作用

   > 迭代器与列表的区别在于，构建迭代器的时候，不像列表把所有元素一次性加载到内存，而是以一种延迟计算（lazy evaluation）方式返回元素，这正是它的优点。比如列表中含有一千万个整数，需要占超过100M的内存，而迭代器只需要几十个字节的空间。因为它并没有把所有元素装载到内存中，而是等到调用next()方法的时候才返回该元素（按需调用 call by need 的方式，本质上 for 循环就是不断地调用迭代器的next()方法

   由于生成器逐个地产生值，而不是一次性地将所有值加载到内存中，因此它们非常适合处理大型数据集。例如，对于处理大量数据的文件读取操作或海量日志分析，如果一次性将这些数据加载到内存可能会导致内存耗尽。生成器可以逐行读取数据，每次只处理一小部分内容

   比如说我们要读一个很大的文件。一种常规的做法是

   ``` python
   def read_large_file_into_memory(file_name):
       with open(file_name, 'r') as file:
           # 读取整个文件到列表中，每个元素是一行
           lines = file.readlines()
       
       # 返回去除了换行符的行列表
       return [line.strip() for line in lines]
   
   # 使用普通函数处理每一行数据
   lines = read_large_file_into_memory('large_log_file.log')
   for line in lines:
       process(line)  # 处理文件的每一行
   ```

   如果这是一个很大的文件的话肯定是不理想的，因为要一次性把整个文件保存到内存中

   但是用生成器我们就可以读一行才加载一行

   ```python
   def read_large_file(file_name):
       with open(file_name, 'r') as file:
           for line in file:
               yield line.strip()  # 移除行尾换行符并返回该行数据
   
   # 使用生成器处理每一行数据
   for line in read_large_file('large_log_file.log'):
       process(line)  # 处理文件的每一行
   ```

3. 表示无限序列：生成器可以表达无穷的数据流，如无限数列。这对于需要延伸到无限范围的算法或模拟实时数据流的应用程序来说非常有用

### 对不可迭代对象进行迭代的两种错误

让自定义类成为一个迭代器，需要在类中实现魔法函数 `__iter__()` 和 `__next__()`

* 当对不可迭代对象（或者说没有实现上面两个魔法函数的对象）运用 for 迭代的时候会报错

  ```python
  >>> class Foo:
  ...     pass
  ...
  >>> for i in Foo():
  ...     print(i)
  ...
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  TypeError: 'Foo' object is not iterable
  ```

* `__iter__()` 没有返回一个可迭代对象

  ```python
  >>> class Foo:
  ...     def __iter__(self):
  ...     	pass
  ...
  >>> for i in Foo():
  ...     print(i)
  ...
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  TypeError: iter() returned non-iterator of type 'NoneType'
  ```

### 自定义类的迭代器实现

首先我们定义一个可迭代对象

```python
class MyIterable(object):
    def __init__(self):
        self.attr1 = []
        self.attr2 = {}
    def add(self, val):
        pass
    def __iter__(self):
        # return iter(self.attr) # 若attr只是一个可迭代对象的话，也可以直接用它的迭代器
        my_iterator = MyIterator(self)
        return my_iterator # 返回迭代器对象
```

然后再为这个可迭代对象定制一个迭代器对象，里面要实现 `__next__()`

但是 Python 语法规定迭代器自己本身就要是一个可迭代对象，因此还需要实现 `__iter__()`

```python
class MyIterator(object):
    def __init__(self, my_iterable):
        self._my_iterable = my_iterable
        self.curr = 0 # 记录当前访问到的位置
    def __next__(self):
        pass # 迭代器的核心逻辑实现，如何找到下一个元素？
    def __iter__(self):
        return self #迭代器自身正是一个迭代器，所以迭代器的__iter__方法返回自身即可
```

## *生成器 Generator*

### 生成器 & 迭代器的关系

生成器是迭代器的一种，但使用起来更简洁。使用 `yield` 语句可以轻松创建生成器，在每次迭代中产生一个值。当函数执行到 `yield` 时，它会返回一个值并暂停执行，直到再次被调用

生成器有两种形式

* 生成器函数 generator function：不用 return，用 yield 返回就属于生成器函数

* 生成器表达式 generator expression

  ```python
  (expression for item in iterable if condition)
  ```

  ```python
  # 创建一个生成器表达式，用于计算每个数的平方
  squares = (x ** 2 for x in range(10))
  
  # `squares` 现在是一个生成器对象
  print(squares)  # 输出: <generator object <genexpr> at ...>
  
  # 可以通过迭代来使用生成器，例如在 for 循环中
  for square in squares:
      print(square, end=' ')  # 输出: 0 1 4 9 16 25 36 49 64 81
  ```

  与列表推导相比，生成器表达式的主要优势在于：

  - **内存效率**：生成器表达式不会一次性将所有元素加载到内存中。它们产生一个一个的元素，这意味着即使是对于非常大的数据集，它们也能保持低内存占用。
  - **惰性求值**：元素会在需要的时候才生成，这允许表示无限长的序列，并且可以节省计算资源

https://zhuanlan.zhihu.com/p/341439647 

一个使用场景是：有时候在序列或集合内的元素个数非常巨大，若全部制造出来并加入内存，对计算机的压力是巨大的，深度学习对权重进行迭代计算就是一个典型的例子。生成器实现的功能就是可以中断性的间隔生成元素（比如一个mini-batch），这样就不必同时在内存中存在整个数据集合，比如整个dataset。从而节省了大量的空间

函数被调用时会返回一个生成器对象。生成器其实是一种特殊的迭代器，一种更加精简高效的迭代器，它不需要像普通迭代器一样实现`__iter__()`和`__next__()`方法了，只需要一个`yield`关键字。因此在实际中一般都会直接实现生成器

下图是 iterable vs. iterator vs. generator 的关系，来源：https://nvie.com/posts/iterators-vs-generators/

<img src="iterable_iterator_generator.png" width="80%">

下面是 i2dl Dataloader 的 `__iter__()` 的例子，用来实现 load 一个 batch

```python
class Dataloader:
    # ...
    def __iter__(self):
        for index in indexes:
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield batch_to_numpy(combine_batch_dicts(batch))
                batch = []

        if not self.drop_last and len(batch) > 0:  # when drop_last == False and len of remaining both ture, will enter
            yield batch_to_numpy(combine_batch_dicts(batch))
```

### 生成器的 `yield` 和普通 `return` 执行流的区别

```python
# A regular method
def regular_foo():
    index = 0
    return_list = []
    while index < 10:
        return_list.append(index)
        index += 1
    return return_list

for x in regular_foo():
    print(x)
```

上面是一个常规的 for 循环，它的执行流是

1. 调用 `rugular_foo()` 得到一个 `return_list`，要返回这个对象把它加载到内存中
2. 将得到的 list 对象返回给 main 函数的 scope 中
3. 迭代得到的 list 对象，并打印

```python
# A generator
def my_generator():
    index = 0
    while index < 10:
        yield index
        index += 1

for x in my_generator():
    print(x)
```

上面是一个 generator，它的执行流是

<img src="generator_process.png">

1. 首次进入函数，按行执行直到遇到 `yield` 语句
2. 函数的状态被保存到内存中，包括已执行行的指针，用以保存 `yield` 之前的运行状态
3. 将索引（0）的值通过 `yield` 返回给外部范围。此时 x == 0
4. 打印 x 的值，即 0
5. 再次进入函数进行下一次迭代。**最关键的概念**：我们再次进入 `my_generator()` 函数，但此时与常规函数调用不同，函数状态从 RAM 中重新加载，并且当前位置在 `index += 1` 这一行
6. 开始 while 循环的第二次迭代，并产出值 1
7. 持续这样的循环，直到 `my_generator()` 内部的 while 循环结束，函数返回并完成执行，即不再生成任何其他值

### 控制生成器的迭代

- `next(generator)`：获取生成器的下一个元素

- `send(value)`：发送一个值到生成器中。这个值会成为生成器中 `yield` 表达式的结果然后生成器继续执行，直到下一个 `yield` 表达式或者结束

  当首次启动生成器时（即在生成器函数中的代码还未执行前），只能发送 `None`；否则会抛出 `TypeError` 异常。在生成器启动之后（即至少执行过一次 `next()`），可以发送非 `None` 的值

  ```python
  def my_generator():
      received = yield 1
      print(f'Received: {received}')
      received = yield 2
      print(f'Received: {received}')
  
  gen = my_generator()
  
  # 启动生成器
  value = next(gen)  # 或者 value = gen.send(None)
  print(value)  # 输出: 1
  
  # 给生成器发送值
  value = gen.send('hello')
  print(value)  # 输出: 2
  ```

  在上面的例子中，第一次调用 `next(gen)` 后，生成器返回 1，并在 `yield 1` 处暂停。随后使用 `send('hello')` 继续执行，此时 `'hello'` 被赋值给 `received` 变量，并打印出来。然后生成器继续执行，直到遇到下一个 `yield`

- `throw(type[, value[, traceback]])`：向生成器中抛出一个异常，如果生成器没有捕获该异常，则迭代终止

- `close()`：结束生成器的迭代，，使其在执行下一个 `yield` 表达式时抛出 `StopIteration` 异常。如果在生成器中有未完成的 `try`...`finally` 块，那么 `finally` 块将被执行

  这实际上是告诉生成器不再需要产生值。尽管生成器函数通常会在抵达末尾自行停止，但 `close()` 方法允许在外部显式地停止生成器，特别是在有无限循环的生成器的情况下

  ```python
  gen = my_generator()
  
  print(next(gen))  # 输出: 1
  
  gen.close()  # 关闭生成器
  
  try:
      print(next(gen))
  except StopIteration as e:
      print('Generator closed')  # 输出: Generator closed
  ```

### 典型错误：在 nested function 中使用 yield

````python
class Dataloader():
    # ...
    def __iter__(self):
        def nested_function(x)
              yield x
        indx = 0
        numbers = []
        while indx < 10:
            numbers.append(nested_function(indx))
            indx += 1 
        return iter(numbers)

dataloader = Dataloader()
for x in dataloader:
    print(x)
````

因为 yield 是从 `nexted_function` 里面返回到了调用的地方，而不是 main，因此在 return 的时候仍然要返回所有数值，这和直接 return 是没有区别的

甚至若连 `__iter__()` 中的 return 都不写的话，main 得到的就只有 None

## *闭包 Closure*

闭包概念是理解装饰器的基础，所以要先看闭包部分

### 闭包的概念

因为 Python 一切皆对象，所以可以在函数内再定义函数，因为函数也是一个类，所以是相当于定义了一个内部类

要创建一个闭包，需要满足以下几个条件

* 在一个外函数中定义了一个内函数，内函数里引用了外函数的临时变量
* 外函数的返回值是内函数的引用 / 指针（C++ 的说法）

这种程序结构的主要作用是：使得函数中的局部变量可以常驻内存，即使在函数返回之后（函数生命期结束后）。在这个意义上它的作用与 C++ 中的 static 静态变量类似，当然不完全相同

### 例子

```python
# outer是外部函数 a和b都是外函数的临时变量
def outer(a):
    b = 10
    # inner是内函数
    def inner():
        print(a + b) # 在内函数中 用到了外函数的临时变量
    return inner # 外函数的返回值是内函数的引用
```

call outer 函数的结果如下：

```python
>>> demo = outer(5)
>>> demo
<function outer.<locals>.inner at 0x7f9455f494c0>
>>> demo()
15
>>> demo2 = outer(7)
>>> demo2()
17
```

`outer(5)` 是 call 了一次 outer 函数，返回了一个 inner 函数对象给 demo，可以看到输入 demo 的时候显示是一个函数对象 inner

然后 call 函数对象 demo，也就是 call inner 的时候，就可以得到返回值 17

从中我们可以看出闭包的一个重要执行顺序性质是：**外函数指针被创建时，内函数未被执行，直到使用函数指针调用内函数才会被执行**

### 外函数把临时变量绑定给内函数

但这里有个很明显的问题，当 outer 调用完毕后，它的临时变量 `b = 10` 应该被系统回收了，可是显然 `demo()` 能得到返回值就说明这个临时变量没有被销毁

一般情况下函数调用后确实会进行资源回收，可闭包是一个例外。一旦外部函数发现自己的临时变量在将来可能会被内部函数用到时，就会把外函数的临时变量交给内函数来管理

注意：每次调用外函数，都返回不同的实例对象的引用，他们的功能是一样的，但是它们实际上不是同一个函数对象，比如demo和demo2就是两个函数对象

### 闭包参数

* 闭包参数放在外函数

  ```python
  def outer(arg):
      temp = 10
      def inner():
          nonlocal  temp  #用nonlocal声明变量，表示要到上一层变量空间寻找该变量
          _sum = temp + arg
          temp += 1   #此处修改temp的值，不会报错
          print('_sum = ', _sum)
          return _sum
      return inner
  
  f = outer(2)
  x = f()
  print('x = ', x) # 12
  x = f()
  print('x = ', x) # 13
  ```

  因为参数在创建函数指针时传入，那么该参数在之后的调用中都会保持原值

* 闭包参数放在内函数

  ```python
  def outer():
      temp = 10
      def inner(arg):
          _sum = temp + arg
          print('_sum = ', _sum)
          return _sum
      return inner
  
  f = outer()
  x = f(2) # 12
  print('x = ', x)
  x = f(5) # 15
  print('x = ', x)
  ```

* 内外函数都有

  ```python
  def outer(prefix):
      def inner(suffix):  # suffix 是内部函数的参数
          print(prefix + ":" + suffix)  # 内部函数既使用外部函数的参数，也使用自己的参数
      return inner
  
  closure = outer("Error")
  closure("404 Not Found")  # 输出: Error:404 Not Found
  ```

  在这个示例中，外部函数 `outer` 接受一个参数 `prefix`，而内部函数 `inner` 接受另一个参数 `suffix`。当 `outer` 被调用并传入 `prefix` 时，它返回闭包 `inner`，后者在调用时需要一个 `suffix` 参数。闭包记住了 `prefix` 变量，并且每次调用闭包时都可以接收新的 `suffix` 值来生成输出

### 闭包陷阱

* 在闭包中内函数修改外函数的局部变量：在[Python 的全局 & 局部变量陷阱](#Python 的全局 & 局部变量陷阱)已经举过一个闭包的例子了，要使用 nonlocal 来声明内函数的局部变量
* 使用闭包时，不要返回任何循环变量或后续会发生变化的变量

### 闭包的作用

* 闭包常被用于创建具有私有变量的函数，这些私有变量无法从外部直接访问，只能通过闭包提供的方法访问。这类似于面向对象编程中的封装

  例如，可以使用闭包来创建计数器、工厂函数等

  ```python
  def create_counter():
      count = 0
  
      def counter():
          nonlocal count
          count += 1
          return count
  
      return counter
  
  counterA = create_counter()
  print(counterA())  # 输出: 1
  print(counterA())  # 输出: 2
  
  counterB = create_counter()
  print(counterB())  # 输出: 1
  ```

  每次调用 `create_counter` 时，都会创建一个新的闭包实例，其中的 `count` 变量是独立的。`counterA` 和 `counterB` 每个都有自己的私有 `count`，并且相互之间不会产生影响

* 装饰器

## *装饰器 Decorator*

类是可以通过继承来扩展类的功能的，但是若想要扩展函数的功能改怎么办呢？利用基于开放封闭原则的闭包来扩展函数

[彻底理解Python中的闭包和装饰器（下） - MidoQ - 博客园 (cnblogs.com)](https://www.cnblogs.com/midoq/p/16961810.html)

### 装饰器原理

装饰器的作用是在不修改函数定义的前提下增加现有函数的功能，比如打印函数名称、计算函数运行时间等。装饰器的本质是一个闭包

<img src="装饰器.drawio.png">

```python
>>> def outer(func):
...     def inner():
...         print("认证成功")
...         result = func()
...         print("日志添加成功")
...         return result
...     return inner
...
>>> @outer
... def f1():
...     print("业务部门1数据接口")
...
>>> f1()
认证成功
业务部门1数据接口
日志添加成功
```

我们可以认为装饰器的 `@` 语法实际上就是闭包调用的语法糖，即它等价为

```python
outer = outer(f1)
outer()
```

如果把函数看作一个黑盒模型，那么被修饰的函数的功能可以分为输入（参数）和输出（返回值），为了使被修饰的函数的性质保持不变，必须保持这两部分不变

* 函数参数不变：**装饰器必须接收一个函数引用的变量，并在闭包中调用且只调用一次**
* 函数返回值不变：**装饰器的内函数必须返回原函数的返回值**

### 给外层函数使用的参数

# 面向对象

## *类基础*

### 类定义和实例化

```python
class Student:
    classID = 1 #类属性
    classroom = "101"
    
    def __init__(self, name, age): #魔法函数：构造函数
        self.name = name #实例化属性
        self.age = age
        
	def print_age(self): #方法
    	print(self.age)
 
li = Student("Ming Li", 10) #实例化
```

### 访问限定

* 若要将属性设置为私有，就直接将属性名用双前导下划线命名（单前导下划线只是建议私有！），即 `__Var`
* 这是通过名称修饰 Name mangling 来实现的，解释器会将 `__var` 替换为 `_classname__var`。因此非得在类外访问私有的化可以通过 `_classname__Var`在类的外部访问`__Var`变量，但不要这么做
* 相当于Python的私有成员和访问限制机制都不是真正写死的，没有从语法层面彻底限制对私有成员的访问。这一点和常量的尴尬地位很相似

### 属性和方法

* 类中的变量称为属性，类属性默认是公有的

  * 实例属性：实例本身拥有的变量，每个实例变量在内存中都不一样

  * 类属性属于整个类的共有属性，为所有实例类所共享，和C++的静态成员变量一样，类属性采用 `类名.类属性` 的方法进行访问，否则会产生下面的问题

  * 根据Python的动态语言特性，当实例化的类修改其类属性的时候实际上是新建了一个属于该实例的实例属性

    ```python
    >>> li.classroom
    '101'
    >>> id(li.classroom
    >>> id(li.classroom)
    140229149483824
    >>> li.classroom = "102"
    >>> id(li.classroom)
    140229149501552
    ```

* 类方法是类中定义的函数。类方法和普通函数的区别是它的第一个参数必须是self，来代表**实例化**的类对象。和C++类方法隐含的this指针一样，不过Python在方法参数中必须显示给出

  ```python
  class Student:
      classID = 1 #类属性
      classroom = "101"
      
      def __init__(self, name, age): #魔法函数：构造函数
          self.name = name #实例化属性
          self.age = age
          
  	def print_age(self): #实例方法
      	print(self.age)
      
      @staticmethod
      def static_method():
          pass
      
      @classmethod
      def class_method(cls):
          pass
  ```

  * 实例方法
    * 由实例调用，至少包含一个self参数，且为第一个参数
    * self的作用和C++中的this指针是一样的，指代的是实例化类。不同于this指针是默认隐藏的，self需要显示给出
    * self不是Python关键字，换成其他的词也可以，但不要这么做
  * 静态方法
    * 静态方法由类调用，无默认参数，用 `@staticmethod` 来修饰
    * 使用 `类名.静态方法` 调用，不建议使用 `实例.静态方法` 的方式调用
    * 静态方法不能获取类属性、构造函数定义的变量，属于 function 类型
    * 静态方法在实际中使用的不多，因为完全可以用一般的函数替代
  * 类方法
    * 类方法由类调用，至少包含一个默认参数cls，用 `@classmethod` 来修饰
    * 使用 `类名.类方法` 调用，不建议使用 `实例.类方法` 的方式调用
    * 可以获取类属性，不能获取构造函数定义的变量，属于 method 类型

### 类在内存中的保存

<img src="类在内存中的保存.png" width="50%">

和C++类的内存模型一样，Python的类、类的所有方法以及类变量在内存中只有一份，所有的实例共享它们。而每一个实例都在内存中独立的保存自己和自己的实例变量

创建实例时，实例中除了封装实例变量之外，还会保存一个类对象指针，该值指向实例所属的类的地址。因此，实例可以寻找到自己的类，并进行相关调用，而类无法寻找到自己的某个实例

## *魔法函数*

Python会为实例化的类配备大量默认的魔法函数 Magic method/dunder method。类似于C++的合成/默认函数

这里只列出最重要的一些。有些魔法函数是自动生成的，有些则需要自己定义

### 默认生成

* `__class__`： 输出类名称

* `__module__`：输出所属的模块名

* `__file__`：当前.py文件的绝对路径

* `__doc__`：输出说明性文档和信息

  ```python
  class A:
      '''
      Documentation 类的说明
      '''
      pass
  print(A.__doc__)
  ```

  三引号的注释方式是专门为 `__doc__` 提供文档内容的。这类注释必须紧跟在定义体下方，不能在任意位置

* `__del__()` 析构函数

  * 当对象在内存中被释放时，会自动触发析构函数
  * 使用 `del` 来进行显式强制删除
  * 因为Python中没有C++中内置类型和自定义类型的设计缺陷，Python自带内存分配和释放机制，所以析构函数一般都不需要自己定义，除非是想在析构时候完成一些动作，比如日志记录等
  
* `__dict__`：列出类或对象中的所有成员。非常重要和有用的一个属性，Python默认生成，无需用户自定义

### 需要自定义

* `__init__()` 构造函数初始化实例属性，`__init__()` 会被默认生成，若用户不显示给出的话，但一般都是要自己给出的

  ```python
  class A:
      def __init__(self, a, b):
          self._a = a
          self._b = b
  ```

  Python 中并没有像 C++ 那样的初始化列表 `{}`。C++ 的初始化列表，Python中统一使用 `()` 来初始化，比如说 `dict()`

  Python 类还有一个 `__new__` 的特殊方法，它是实际的构造器，负责返回一个新的类实例，在大多数情况下，你不需要重写它，除非你有特殊的需求，比如控制创建实例的过程

* `__call__()` 令类成为一个可调用对象，相当于C++中的 `operator()` 仿函数。如果为一个类编写了该方法，那么在该类的实例后面加括号，可会调用这个方法，即 `对象()` 或者 `类()()`。这个魔法函数很重要，基本上都要自定义实现

  ```python
  def __call__(self, *args, **kwargs):
      print('__call__')
  ```

  可以利用内建的 `callable()` 函数来判断一个对象是否可以call

* `__iter__()` 是实例化类对象称为一个可迭代对象，这在前面的迭代器部分有讲解

* `__len__()` 获取对象长度

* `__str__() `与 `__repr__`

  * 当 `print(obj)` 时会调用对象的 `__str__()`魔法函数，输出自定义的返回值。这个魔法函数很重要，通常都需要自定义
  * `__repr__()` 与 `__str__()` 在于前者是给调试人员看的，后者是给普通用户看的

* `__getitem__(), __setitem__(), __delitem__()` 设置 `[ ]` 的取值操作，和 property 修饰器相似，分别表示取值、赋值和删除的作用

  ```python
  class Foo:
      def __getitem__(self, key):
          pass
      def __setitem__(self, key, val):
          pass
      def __delitem__(self, key):
          pass
  
  obj = Foo()
  res = obj['key'] #get
  obj['key'] = 3 #set
  del obj['key'] #del
  ```

* 运算符重载

  * `__add__`: 加运算
  * `__sub__`: 减运算 
  * `__mul__`: 乘运算 
  * `__div__`: 除运算 
  * `__mod__`: 求余运算 
  * `__pow__`: 幂运算

## *继承与多态*

### 定义

```python
class Foo(superA, superB, superC....):
class DerivedClassName(modname.BaseClassName):  # 当父类定义在另外的模块时
```

Python支持多继承，按照括号里父类的继承顺序依次继承，在Python中父类又称为超类 Superclass

**所有类的最终父类都是 object**，即使是没有显示继承

### Python3的继承机制

Python3的继承机制不同于Python2，核心原则是

* 子类在调用某个方法或变量的时候，首先在自己内部查找；若没有找到，则根据继承顺序依次在父类找

* 多继承：根据父类继承顺序，先去找括号内最左边（也就是最先继承）的，如果有被查找对象，则最先调用，采用的是深度优先搜索调用。当一条路走到黑也没找到的时候，才换右边的另一条路

  <img src="多继承的DFS搜寻顺序.drawio.png" width="20%">

  上图中A是基类，D和G分别是两条继承路线的父类

* 菱形继承

  <img src="菱形继承搜索顺序.drawio.png" width="20%">

### `super()`

父类和子类中有同名成员时，子类成员将直接屏蔽对父类中同名成员的直接访问，这种情况叫做隐藏 hide，也叫做重定义。C++中若在子类中想要访问父类中被隐藏的成员，可以通过指定父类域的方式，即 `父类::父类成员` 显式访问

Python中则会通过 `super()` 函数从子类中强制调用父类的被隐藏函数，最常见的就是通过 `super()` 来调用父类的构造函数

```python
class A:
    def __init__(self, name):
        self.name = name
        print("父类的__init__方法被执行了！")
    def show(self):
        print("父类的show方法被执行了！")

class B(A):
    def __init__(self, name, age):
        super(B, self).__init__(name=name)
        self.age = age

    def show(self):
        super(B, self).show()
```

`super(子类名, self).方法名()` 需要传入的是子类名和self，调用的是父类里的方法，按夫类的方法需要传入参数

### 多态

Python的多态与重写非常简单，不需要像C++那样要满足2个必要条件（必须通过父类的指针或者引用调用虚函数和被调用的函数必须是虚函数，且子类必须对父类的虚函数进行重写）。只要直接在子类中实现一个同名函数就行了

这可能是因为Python中没有函数重载体系，相同名字的函数，即使是参数不同也会被覆盖

### 接口类

Python没有像C++纯虚类或Java中的显式接口 interface 关键字，因为它是一种动态类型语言。不过可以通过抽象基类（Abstract Base Class，ABC）来模拟接口类的行为。在Python中，可以使用`abc`模块来定义一个抽象基类，并使用装饰器`@abstractmethod`标记那些在子类中必须被重写的方法

下面是一个例子，展示了如何定义一个包含抽象方法的接口类：

```python
from abc import ABC, abstractmethod

class MyInterface(ABC):
    
    @abstractmethod
    def my_method(self):
        pass

class ConcreteClass(MyInterface):
    
    def my_method(self):
        print("实现了接口中的my_method")

# 如果尝试实例化MyInterface，将会抛出TypeError，
# 因为它包含有抽象方法而不能被实例化。
# my_interface = MyInterface()  # TypeError: Can't instantiate abstract class MyInterface with abstract method my_method

# 只有当所有的抽象方法都被覆盖时，子类才能被实例化。
concrete_class = ConcreteClass()
concrete_class.my_method()  # 输出: 实现了接口中的my_method
```

如果创建了一个 `MyInterface` 的子类但没有实现所有的抽象方法，Python 解释器将不允许实例化该子类。这就迫使任何继承自 `MyInterface` 的子类提供`my_method`方法的具体实现

## *Property decorator*

Python 内置的 `@property` 装饰器可以**把类的方法伪装伪装成属性调用的方式**，也就是本来 `Foo.func()` 的调用方式变成了 `Foo.func` 的方式

将一个方法伪装成属性后，就不能再使用圆括号的调用方式了

### 设置装饰器的步骤

```python
class People:
    def __init__(self, name, age):
        self.__name = name
        self.__age = age
   	@property
    def age(self):
        return self.__age
    @age.setter
    def age(self, age):
        if isinstance(age, int):
            self.__age = age
        else:
            raise ValueError
    @age.deleter
    def age(self):
        print("Delete age")
obj = People("Zhangsan", 18) #Instantiation
print(obj.age)
obj.age = 20 #Modify
del obj.age #Delete
```

以下三个修饰器分别对应与对一个属性的获取、修改和删除

* 在普通方法的基础的基础上添加 `@property` 装饰器，例如上面的 `age()` 方法。相当于是一个get方法，用来获取值
* 写一个同名方法，并添加 `@xxx.setter`（xxx表示和修饰方法一样的名字）装饰器，这相当于编写了一个set方法，提供赋值功能
* 再写一个同名的方法，并添加 `@xxx.deleter` 修饰器，这相当于写了一个删除功能

还可以定义只读属性，也就是只定义 getter 方法，不定义 setter 方法就是一个只读属性

### property 函数

Python内置的 `@property` 装饰器可以把类的方法伪装成属性调用的方式。也就是本来是`Foo.func()`的调用方法，变成`Foo.func`的方式

```python
def People():
    # ...
    def get_age(self):
        pass
    def set_age(self, age):
        pass
    def del_age(self):
        pass
    age = property(get_age, set_age, del_age, "年龄")
```

除了装饰器外，还可以使用Python内置的builtins模块中的property函数来设置

## *特殊类*

### 枚举类

枚举在 Python 中是通过标准库中的 `enum` 模块实现的

创建一个枚举很简单，只需继承自 `Enum` 类，并为其添加一些类属性即可

```python
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
```

枚举成员有两个主要的属性：`name` 和 `value`。`name` 属性返回成员的名称，`value` 属性返回成员的值

```python
print(favorite_color.name)  # 输出: 'RED'
print(favorite_color.value)  # 输出: 1
```

如果不关心枚举成员的具体数值，可以使用 `auto()` 函数自动为其分配值

```python
from enum import auto

class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()
```

Python的枚举提供了大量的定制化方法，例如可以限制枚举的修改，或者创建值相同的别名。此外，还有其他类型的枚举，如 `IntEnum`, `Flag`, `IntFlag` 等，它们提供了额外的功能

# IO

## *输入*

### input

`input` 函数用于从标准输入（通常是键盘）读取用户输入的数据。当 `input` 函数被调用时，程序会暂停执行，等待用户输入文本并按下回车键

```python
variable = input(prompt)
```

- `prompt` 是一个字符串，用作提示用户输入的文本。这个参数是可选的，如果不提供，用户将不会看到任何提示信息
- 当用户完成输入并按下回车键后，输入的文本将作为字符串返回，并可以被赋值给变量

## *文件*

### 打开文件

用 `open(filename, mode)` 打开一个文件，返回一个文件描述符对象，打开文件的模式和C的IO比较相似，可以看下面这张图

<img src="openMode.png" width="60%">

使用文件完毕后，需要用文件方法 `close()` 来关闭，但是比起这种方法，经常会使用更便捷的with来处理。在使用结束后，它会帮助用户正确的关闭文件，不需要再进行其他操作

```python
with open('filename', 'r') as f:
     read_data = f.read()
```

### with-as关键字

[Python中的with-as用法 - 简书 (jianshu.com)](https://www.jianshu.com/p/c00df845323c)

`with-as` 语句在Python中通常用于简化异常处理和清理代码，尤其是在处理资源时。它允许你包装代码块的执行，以便资源的分配和释放可以被自动管理。这种做法在Python中通常称为上下文管理器 context manager

```python
with expression as variable:
    with_block
```

- `expression` 部分通常是一个会返回上下文管理器对象的表达式。上下文管理器对象需要实现特定的方法 (`__enter__` 和 `__exit__`) 来定义在代码块开始执行前后应该发生什么
- `as variable` 部分是可选的，如果存在，它将把 `__enter__` 方法的返回值赋予变量，这个变量通常用于`with`块内
- `with_block` 是需要执行的代码块，在这个块内部可以使用由`as variable`定义的变量

一个常见的常见就是上面的打开文件，相比于C/C++那种用open打开文件，获得一个文件句柄进行操作，用完后还要关闭它，Python直接用with-as就行了

在上面的例子中`open()` 函数返回了一个文件对象，这个对象作为上下文管理器用来保证无论如何都能安全地关闭文件。当进入`with`代码块时，文件会被打开，并且`file`变量会引用这个文件对象。当离开`with`代码块时，无论是正常退出还是因为异常而退出，`__exit__`方法会被调用，此时文件会被自动关闭

### 几种文件读取方法的对比

* `f.read()`

  ```python
  with open('testRead.txt', 'r', encoding='UTF-8') as f1:
      results = f1.read()
      print(results)
  ```

  读取整个文件，返回的是一个字符串，字符串包括文件中的所所有内容。若想要将每一行数据分离，即需要对每一行数据进行操作，此方法无效

  `read()` 是最快的，当然其功能最简单，在很多情况下不能满足需求

* `f.readline()` 会从文件中读取单独的一行。换行符为 '\n'。`f.readline()` 如果返回一个空字符串, 说明已经已经读取到最后一行

  ```python
  with open('testRead.txt', 'r', encoding='UTF-8') as f2:
      line = f2.readline()
      while line is not None and line != '':
          print(line)
          line = f2.readline()
  ```

  每次只读取文件的一行，内存占用低

* `f.readlines()`

  ```python
  with open('testRead.txt', 'r', encoding='UTF-8') as f3:
      lines = f3.readlines()    # 接收数据
      for line in lines:     # 遍历数据
          print(line)
  ```

  不推荐用于大文件，因为它会一次性将整个文件内容读取到内存中；对于小文件来说很快，因为只进行了一次磁盘 IO 操作

* 使用文件对象迭代

  ```python
  with open('file.txt', 'r') as file:
      for line in file:
          # 处理每一行...
  ```

  直接对文件对象进行迭代，逐行读取。推荐用于任何大小的文件。Python会为文件内容创建一个迭代器，并且一次只处理一行数据，所以内存占用少。这种方法类似于 `readline()`

### 其他 File 方法

假设有 `f=open(filename, mode)`

* `f.write()`

* `f.tell()` 返回文件对象当前所处的位置, 它是从文件开头开始算起的字节数

* `f.seek()` 改变文件指针当前的位置

## *os文件/目录方法*

os 模块提供了非常丰富的方法用来处理文件和目录

```python
import os
```

### 操作目录

`__file__` 是当前.py文件的绝对路径

* 获取当前目录

  ```python
  current_directory = os.getcwd()
  ```

* 改变当前目录

  ```python
  os.chdir('/path/to/your/directory')
  ```

* 列出目录内容

  ```python
  entries = os.listdir('/path/to/your/directory')
  print(entries)
  ```

* 创建目录

  ```python
  os.mkdir('/path/to/new/directory')
  ```

  递归创建目录

  ```python
  os.makedirs('/path/to/new/directory/with/subdirectory', exist_ok=True)
  ```

### 操作路径

`os.path` 模块用于对路径的各种操作

* 检查路径是否存在

  ```python
  is_exist = os.path.exists('/path/to/check')
  ```

* 检查路径类型

  ```python
  is_file = os.path.isfile('/path/to/check')
  is_dir = os.path.isdir('/path/to/check')
  ```

* 路径拼接

  ```python
  full_path = os.path.join('directory', 'subdirectory', 'file.txt')
  ```

* 分解路径

  ```python
  head, tail = os.path.split('/path/to/split/file.txt')
  ```

* 获取文件名和扩展名

  ```python
  basename = os.path.basename('/path/to/some/file.txt')  # 返回 'file.txt'
  dirname = os.path.dirname('/path/to/some/file.txt')    # 返回 '/path/to/some'
  name, extension = os.path.splitext('/path/to/some/file.txt')  # 返回 ('/path/to/some/file', '.txt')
  ```

* 获取绝对路径

  ```python
  absolute_path = os.path.abspath('relative/path/to/file')
  ```

### 环境变量

`environ`是一个代表当前环境变量的字典对象。环境变量通常包含了系统运行时的配置信息，比如用户的主目录、执行路径（PATH）以及操作系统类型等。

通过`os.environ`可以访问和修改这些环境变量。因为它是一个字典，所以可以用所有标准的字典操作对其进行读取、修改、添加或删除环境变量

```python
import os

# 获取名为'HOME'的环境变量值
home_directory = os.environ.get('HOME')
print(home_directory)

# 设置一个新的环境变量，或者修改一个已有的环境变量
os.environ['NEW_VAR'] = 'SomeValue'

# 打印出所有的环境变量及其值
for key, value in os.environ.items():
    print(f"{key}: {value}")
```

### 与系统相关的其他方法

* `os.times()`：返回当前的全局进程时间，五个属性
* `os.sep`：不同OS的分隔符是不同的，Win文件的路径分隔符是 `'\'`，在Linux上是 `'/'`，可以用 `os.sep` 来代替

## *shutil*

shutil (shell utilities) 模块提供了一系列对文件和文件集合进行高阶操作的函数

`shutil.move()` 可以用于移动文件或重命名文件

`shutil.copytree()` 可以递归地复制整个目录树

## *sys库*

# 并发

## *进程*

[subprocess Python执行系统命令最优选模块 - 金色旭光 - 博客园 (cnblogs.com)](https://www.cnblogs.com/goldsunshine/p/17558075.html#gallery-1)

subprocess 模块允许我们启动一个新进程，并连接到它们的输入/输出/错误管道，从而获取返回值

Subprocess模块开发之前，标准库已有大量用于执行系统级别命令的的方法，比如os.system、os.spawn等。但是略显混乱使开发者难以抉择，因此subprocess的目的是打造一个统一模块来替换之前执行系统界别命令的方法。所以推荐使用subprocess替代了一些老的方法，比如：os.system、os.spawn等

<img src="subprocess模块.png">

### run 方法

Popen 是 subprocess 的核心，子进程的创建和管理都靠它处理

```python
subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None,
               capture_output=False, shell=False, cwd=None, timeout=None,
               check=False, encoding=None, errors=None, text=None, env=None, universal_newlines=None)
```

- args：这个参数可以是一个字符串，或者是一个由程序参数组成的列表。通常建议使用列表形式，因为这样可以避免 shell 的解析问题，尤其是在参数中包含空格、引号或其他特殊字符时。如果传递一个字符串，并且需要 shell 来解析命令中的管道、文件重定向等 shell 特性，那么必须将 `shell=True` 作为 `Popen` 的参数之一

  当使用列表形式时，第一个列表元素是要运行的命令（例如 `ls`, `grep`, `echo` 等），之后的列表元素是传递给该命令的参数

  比如 `["-p", "compile_commands.json"]` 这样的列表

- stdin、stdout 和 stderr：子进程的标准输入、输出和错误。其值可以是 subprocess.PIPE、subprocess.DEVNULL、一个已经存在的文件描述符、已经打开的文件对象或者 None。subprocess.PIPE 表示为子进程创建新的管道。subprocess.DEVNULL 表示使用 os.devnull。默认使用的是 None，即系统的标准输出和标准错误流，如果特别指定了就可以对其进行操作。另外，stderr 可以合并到 stdout 里一起输出

- timeout：设置命令超时时间。如果命令执行时间超时，子进程将被杀死，并弹出 TimeoutExpired 异常

- check：如果该参数设置为 True，并且进程退出状态码不是 0，则弹 出 CalledProcessError 异常

- encoding: 如果指定了该参数，则 stdin、stdout 和 stderr 可以接收字符串数据，并以该编码方式编码。否则只接收 bytes 类型的数据

- shell：如果该参数为 True，将新建一个 shell 来执行指定的命令

### Popen

run 方法调用方式返回 CompletedProcess 实例，和直接 Popen 差不多，实现是一样的，实际也是调用 Popen。Popen相比于run方法提供了更精细的控制

run 和 popen最大的区别在于：run方法是阻塞调用，会一直等待命令执行完成或失败；popen是非阻塞调用，执行之后立刻返回，结果通过返回对象获取

### 结束程序

1. `exit()` 函数: `exit()` 是一个内置函数，当调用时将抛出一个 `SystemExit` 异常，因此它会退出当前正在运行的程序。这个函数主要应该在交互式解释器中使用

   ```python
   exit()  # 在交互式解释器中退出
   ```

2. `sys.exit()` 函数: `sys.exit()` 函数同样抛出 `SystemExit` 异常，并且允许你退出程序并提供一个可选的退出状态码，它通常用于程序文件

   ```python
   import sys
   
   sys.exit()        # 退出程序，退出状态为 0（默认）
   sys.exit(0)       # 显式地以状态 0 退出（表示成功）
   sys.exit(1)       # 退出程序，指定退出状态 1（通常表示失败）
   ```

3. 抛出 `SystemExit` 异常: 直接抛出 `SystemExit` 异常也会导致程序退出

   ```python
   raise SystemExit
   ```

4. `os._exit()` 函数: 模块 `os` 提供了 `_exit()` 函数，直接终止进程，不会抛出异常，也不会执行任何清理操作（例如关闭文件），因此它比 `sys.exit()` 更加粗暴。它接收一个退出状态码作为参数

   ```python
   import os
   
   os._exit(0)  # 立即退出程序，状态为 0
   ```

注意：尽管 `exit()` 和 `sys.exit()` 功能相似，但 `sys.exit()` 通常在脚本和程序中使用，因为 `exit()` 主要是为 Python 解释器设计的。另外，`sys.exit()` 可以通过 `try...except` 结构捕获 `SystemExit` 异常进行处理，而 `os._exit()` 由于不抛出异常，无法被捕获处理

最后，选择哪种方式取决于具体场景和希望程序如何结束。**在大多数情况下，推荐使用 `sys.exit(ErrorCode)` 来退出程序**

## *线程*

Python的设计者认为操作系统本身的线程调度已经非常成熟稳定了，没有必要自己搞一套。所以Python的线程就是C语言的一个pthread，并通过操作系统调度算法进行调度（例如linux是CFS）

### threading

Python 的多线程是通过内置的 thread 和 threading 模块来实现的。thread提供了低级别的、原始的线程以及一个简单的锁，而threading高级模块提供其他方法，是对thread的进一步封装，大部分情况下都是直接用threading模块

```python
import threading
def thread_function(name):
    print(f"Thread {name} starting")

my_thread = threading.Thread(target=thread_function, args=("MyThread",))

my_thread.start()
# ...
my_thread.join()
```

### TLS

Python中TLS的管理是由 `threading.local` 来实现的。这意味着该数据结构中的数据对于每个线程来说都是唯一的，不同的线程不能访问其他线程中的数据。这可以用来避免在多线程环境中共享资源时出现同步问题。

```python
import threading

# 创建一个全局的threading.local对象
mydata = threading.local()

def process_data():
    # 输出当前线程关联的mydata.x的值，如果没有，则设为0
    print(f"Initial value in thread {threading.current_thread().name}: {getattr(mydata, 'x', 0)}")
    
    mydata.x = 0
    for _ in range(5):
        mydata.x += 1  # 在当前线程内计数
    # 输出当前线程与最终mydata.x的值
    print(f"Final value in thread {threading.current_thread().name}: {mydata.x}")

# 创建并启动两个线程
t1 = threading.Thread(target=process_data, name="Thread-A")
t2 = threading.Thread(target=process_data, name="Thread-B")

t1.start()
t2.start()

t1.join()
t2.join()
```

## *GIL*

Global Interpreter Lock, GIL 全局解释器锁。这是一种线程管理机制，并不属于Python语言的一个整体特性，而是存在于CPython实现的解释器中

GIL是Python解释器设计的历史遗留问题，通常我们用的解释器是官方实现的CPython，要真正利用多核，除非重写一个不带GIL的解释器

> In CPython, the global interpreter lock, or GIL, is a mutex that prevents multiple native threads from executing Python bytecodes at once. This lock is necessary mainly because CPython’s memory management is not thread-safe. (However, since the GIL exists, other features have grown to depend on the guarantees that it enforces.

GIL 是 Python 解释器中的一个技术，它确保任何时候只有一个线程在执行 Python 字节码。Python内部会计算执行当前线程执行字节码的数量，当达到了一定阈值后就强制释放GIL。这意味着即便在多核处理器上，Python 程序的单个进程内部也无法实现真正的并行计算。Python的多线程只适合于IO密集型的程序使用

- 在 IO 密集型应用（比如网络交互、文件操作）中，由于IO线程经常处于等待状态，它可以释放掉GIL，让其他线程执行。因此多线程可以显著提高程序性能，因为线程可以在不占用 CPU 执行时间的情况下完成工作
- 在某些操作延迟较长的任务中，多线程可以改善用户界面的响应性，例如 GUI 应用程序

移除GIL的一些最新进展：[Python团队官宣下线GIL：可选择性关闭 | 量子位 (qbitai.com)](https://www.qbitai.com/2023/07/72584.html)

### 使用多进程来替代多线程

multiprocessing库的出现很大程度上是为了弥补thread库因为GIL而低效的缺陷。它完整的复制了一套thread所提供的接口方便迁移。唯一的不同就是它使用了多进程而不是多线程。每个进程有自己的独立的GIL，因此也不会出现进程之间的GIL争抢

当然multiprocessing也不是万能良药。它的引入会增加程序实现时线程间数据通讯和同步的困难

## *异步IO*

## *协程*

### Python协程的发展历史

1. Python2.5 为生成器引用 `.send()`、`.throw()`、`.close()` 方法
2. Python3.3 为引入yield from，可以接收返回值，可以使用yield from定义协程
3. Python3.4 加入了asyncio模块
4. Python3.5 增加async、await关键字，在语法层面的提供支持
5. Python3.7 使用async def + await的方式定义协程
6. 此后asyncio模块更加完善和稳定，对底层的API进行的封装和扩展
7. Python将于3.10版本中移除以yield from的方式定义协程

# 网络编程

## *socket*

`socket` 库提供了基本的 BSD 套接字接口。它是 Python 网络编程的底层库，支持 TCP 和 UDP 协议，并允许你实现客户端和服务器应用程序。

```python
import socket

# 创建一个 socket 对象
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 连接到服务器
s.connect(('example.com', 80))

# 发送数据
s.sendall(b'GET / HTTP/1.1\r\nHost: example.com\r\n\r\n')

# 接收响应数据
data = s.recv(1024)

print(data.decode())

# 关闭连接
s.close()
```



## *http*

### Server

`http.server` 模块可以用来快速创建一个简易的 HTTP 服务器。这通常用于测试或本地开发阶段，而不推荐在生产环境中使用

```python
from http.server import HTTPServer, BaseHTTPRequestHandler

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Hello, World!')

if __name__ == '__main__':
    httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
    print("Serving at port 8000")
    httpd.serve_forever()
```

### Cient

`http.client` 是一个用于 HTTP 客户端的库，它提供了一些类和方法来发送 HTTP 请求和接收 HTTP 响应

```python
import http.client

# 创建一个 HTTPConnection 对象
conn = http.client.HTTPSConnection("www.example.com")

# 发送 GET 请求
conn.request("GET", "/")

# 获取响应
resp = conn.getresponse()
print(resp.status, resp.reason)

# 读取响应内容
data = resp.read()
print(data)

# 关闭连接
conn.close()
```



## *urllib*

## *ssl*

# Web 开发

见 *Flask.md*

# 调用程序

## *eval & exec*

`eval()` 和 `exec()` 都是Python内置的、用于动态执行代码的函数，exec的功能更强大一些。都定义在builtins.pyi中

### eval

```python
def eval(
    __source: Union[str, bytes, CodeType], __globals: Optional[Dict[str, Any]] = ..., __locals: Optional[Mapping[str, Any]] = ...
```

`eval()` 可以用来动态地计算数学表达式，或者从字符串中解析出数据结构

```python
# 用 eval() 来计算数学表达式
expression = '3 * (4 + 5)'
result = eval(expression)
print(result)  # 输出: 27

# 用 eval() 解析字符串为列表
list_str = '[1, 2, 3, 4, 5]'
my_list = eval(list_str)
print(my_list)  # 输出: [1, 2, 3, 4, 5]
```

### exec

```python
def exec(
    __source: Union[str, bytes, CodeType],
    __globals: Optional[Dict[str, Any]] = ...,
    __locals: Optional[Mapping[str, Any]] = ...,
) -> Any: ...

# Execute the given source in the context of globals and locals.
```

`exec()` 可以用来执行较为复杂的代码片段，例如定义变量、执行循环等操作

```python
# 定义一个Python代码字符串
code = """
def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result

# 计算阶乘并打印结果
fact_5 = factorial(5)
print('The factorial of 5 is:', fact_5)
"""

# 使用 exec() 执行定义的代码
exec(code)
```

### 二者的区别

1. **用途**：
   - `eval()` 用于计算单个Python表达式的值并返回结果。它主要用于数学运算或字典/列表解析等。
   - `exec()` 用于执行动态创建的程序代码，可以是一段复杂的代码块，包括定义变量、循环、类和函数定义等，并不返回任何结果。
2. **返回值**：
   - `eval()` 执行Python表达式并返回表达式的值。
   - `exec()` 不返回任何值，即返回值为 `None`。
3. **输入**：
   - `eval()` 只能接受单个表达式作为输入，比如 `'3 + 5'` 或者 `'"Hello " + "World!"'` 等。
   - `exec()` 可以执行复杂的Python代码，包括包含多条语句的字符串。
4. **安全性**：
   - `eval()` 的风险在于，如果它处理的表达式来自不可信的源，可能会导致意料之外的行为，甚至安全问题。因此，使用时需要特别小心。
   - `exec()` 同样存在安全风险，因为它能够执行更复杂的代码块，恶意代码的危害性也更大。

```python
# 使用 eval()
result = eval('3 * 5')
print(result)  # 输出：15

# 使用 exec()
exec('result = 3 * 5')
print(result)  # 输出：15

# eval() 只能执行表达式
try:
    eval('for i in range(5): print(i)')
except SyntaxError as e:
    print("SyntaxError:", e)

# exec() 可以执行复杂的代码块
exec('for i in range(5): print(i)')
```

## *调用C/C++程序*

###  ctypes库

`ctypes` 是Python的一个标准库，它允许调用C库中的函数。这种方式不需要写额外的C代码，只需要知道目标函数的签名即可

```python
from ctypes import cdll

# 加载动态链接库（假设库文件名为"libexample.so"）
lib = cdll.LoadLibrary('libexample.so')

# 调用其中的函数（假设有一个名为func的函数）
result = lib.func()
```

对于Windows上的`.dll` 或 macOS 的 `.dylib` 文件也可以使用相类似的方法

### Python/C API

Python提供了一套用C语言编写扩展模块的API，这样可以把C或C++编写的代码编译成Python模块，并直接在Python程序中导入使用

```C
#include <Python.h>

static PyObject* my_function(PyObject* self, PyObject* args) {
    // ... 实现功能 ...
    Py_RETURN_NONE;
}

static PyMethodDef MyMethods[] = {
    {"my_function",  my_function, METH_VARARGS, "Description"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mymodule = {
    PyModuleDef_HEAD_INIT,
    "mymodule",
    NULL,
    -1,
    MyMethods
};

PyMODINIT_FUNC PyInit_mymodule(void) {
    return PyModule_Create(&mymodule);
}
```

此代码创建了一个简单的Python模块，其中包含了一个函数`my_function`。编译这段代码生成的模块可以在Python中直接被导入和使用。

### SWIG

SWIG, Simplified Wrapper and Interface Generator 是一个自动化工具，可以将C和C++代码转换为多种高级编程语言的扩展，包括Python

需要写一个SWIG接口文件，该文件定义了想要在Python中使用的C/C++代码的部分

```python
/* example.i */
%module example
%{
#include "example.h"
%}

int some_function(int arg);
// 其他需要暴露给Python的声明...
```

然后，使用SWIG工具生成Python绑定代码，并与原生C/C++代码一起编译为Python扩展模块

### Cython

Cython是一个优化静态编译器，用于将Python代码以及Cython特定语法的代码转换为C代码，并编译为Python扩展模块

```python
# example.pyx
def my_function():
    # ... 实现功能 ...
    pass
```

需要编写一个`.pyx`文件，在里面实现功能，然后使用Cython工具将其编译为C代码，并进一步生成Python模块

### CFFI 

CFFI, C Foreign Function Interface 是另一种可以在Python中调用C代码的工具，它旨在提供一个简单的调用接口，并支持调用被封装的C代码

```python
from cffi import FFI

ffi = FFI()

# 定义C函数的接口
ffi.cdef("""
    int some_function(int x);
""")

# 加载库
C = ffi.dlopen("example.so")

# 调用函数
result = C.some_function(10)
```

# 异常 & 垃圾回收

## *异常*

### 使用异常处理

```python
try:
    # 尝试执行的代码块
    pass
except ExceptionType1:
    # 处理ExceptionType1
    pass
except (ExceptionType2, ExceptionType3) as e:
    # 共同处理ExceptionType2 和 ExceptionType3
    # 可以通过e获取异常实例
    pass
else:
    # 如果没有异常发生，则执行这里的代码
    pass
finally:
    # 无论是否发生异常，最终都会执行的代码块
    pass
```

### 内置异常

<img src="Python异常体系.png">

- `SyntaxError`：代码语法错误
- `IndexError`：列表索引超出范围
- `KeyError`：字典中查找一个不存在的关键字
- `FileNotFoundError`：读取一个不存在的文件
- `ValueError`：传入一个调用者不期望的值，即使该值的类型是正确的
- `TypeError`：操作或函数应用于不适当类型的对象

### 手动抛出异常

```python
raise [Exception [, args [, traceback]]]
```

### 用户自定义异常

```python
class MyCustomError(Exception):
    def __init__(self, message):
        super().__init__(message)
```

可以继承自`Exception`类或其子类来创建自定义的异常类型

# Python编译

### Python 解释器

* CPython：官方版本的解释器，是用C语言开发的，CPython是使用最广的Python解释器，一般用的都是这个解释器
* IPython：基于CPython之上的一个交互式解释器
* PyPy：一个追求执行速度的Python解释器，采用JIT技术，对Python代码进行动态编译
* Jython：运行在Java平台上的Python解释器，可以直接把Python代码编译成Java字节码执行
* IronPython：和Jython类似，只不过IronPython是运行在微软.Net平台上的Python解释器，可直接把Python代码编译成.Net的字节码

### Python 运行机制

<img src="python运行机制.png">



### Python的执行过程

<img src="Python的编译过程.webp" width="60%">

1. 启动解析器

   * 启动Python解释器

     * 加载内建模块和第三方库

     * 设置一些全局变量，比如 `__name__`

2. 编译源文件

   * 解释器读取源文件，然后将其转换成抽象语法树 AST
   * 编译成字节码
   * 编译后的字节码会被保存在对应的 `__pycache__/xxx.pyc` 文件中，如果 `.pyc` 文件存在且 `.py` 源文件没有改变（解释器在转换之前会判断代码文件的修改时间是否与上一次转换后的字节码pyc文件的修改时间一致），则解释器跳过上述两步直接加载 `.pyc` 文件里的字节码来执行

3. 执行：Python虚拟机（PVM）顺序执行（除非遇到控制指令）字节码

4. 垃圾回收和清理退出

   * 通过引用计数对垃圾对象进行析构和内存回收

   * 结束时回收所有Python对象占用的内存，关闭Python虚拟机

   * 返回状态码

   * 结束Python解释器进程
### 编译缓存

.pyc

.pyo

### codeobject

PyCodeObject 就是字节码

## *字节码文件*

# IPython的使用技巧

https://www.zhihu.com/tardis/zm/art/104524108?source_id=1003

IPython是一种基于Python的交互式解释器，提供了强大的编辑和交互功能

通过 `?` 显示对象签名、文档字符串和代码位置；通过 `??` 显示源代码

Python shell不能直接执行shell命令，需要借助sys；IPython通过 `!` 调用系统命令，比如 `! uptime`

# 常用的Python工具包

关于 conda 和 pip 的使用可以看 *包管理工具.md*

## *JSON & Pickle*

Python 的 `json` 模块提供了一种简单的方式来编码和解码JSON数据

```Python
import json
```

### Python数据类型 & JSON数据类型的对应关系

Python `<-->` JSON

- dict `<-->` object
- list, tuple `<-->` array
- str `<-->` string
- int, float, int- & float-derived Enums `<-->` number
- True `<-->` true
- False `<-->` false
- None `<-->` null

### 将 Python 对象编码成 JSON 字符串（序列化）

- `json.dumps()`: 将 Python 对象转换成 JSON 格式的字符串

  ```python
  data = {
      'name': 'John Doe',
      'age': 30,
      'is_employee': True,
      'titles': ['Developer', 'Engineer']
  }
  
  json_string = json.dumps(data)
  print(json_string)  # 输出 JSON 格式的字符串
  ```

- `json.dump()`: 将 Python 对象转换成 JSON 格式的字符串，并将其写入到一个文件中

  ```python
  with open('data.json', 'w') as outfile:
      json.dump(data, outfile)
  # 这会创建 data.json 文件，并写入 JSON 数据
  ```

### 将 JSON 字符串解码成 Python 对象（反序列化）

- `json.loads()`: 将 JSON 格式的字符串解码成 Python 对象

  ```python
  json_string = '{"name": "John Doe", "age": 30, "is_employee": true, "titles": ["Developer", "Engineer"]}'
  
  data = json.loads(json_string)
  print(data)  # 输出解码后的 Python 字典
  ```

- `json.load()`: 读取一个文件，并将其中的 JSON 字符串解码成 Python 对象

  ```python
  with open('data.json', 'r') as infile:
      data = json.load(infile)
  # 从 data.json 读取内容，并转换成 Python 对象
  ```

### 高级选项

`json.dumps()` 和 `json.dump()` 方法接受多个可选参数，以定制编码过程：

- `indent`: 指定缩进级别，用于美化输出。例如，`indent=4` 会用四个空格缩进。
- `separators`: 指定分隔符，默认是 `(', ', ': ')`。如果你想让输出更紧凑，可以使用 `(',', ':')`。
- `sort_keys`: 当设置为 `True` 时，字典的键将被排序。

```python
json_string = json.dumps(data, indent=4, sort_keys=True)
print(json_string)
```

同样，`json.loads()` 和 `json.load()` 也有参数来处理特定情况，比如解析不符合 JSON 规范的数据

### Pickle 包

Python 的 pickle 和 json 一样，两者都有 dumps、dump、loads、load 四个API

* json 包用于字符串和 Python 数据类型间进行转换
* pickle 用于 Python 特有的类型和 Python 的数据类型间进行转换

json 是可以在不同语言之间交换数据的，而 pickle 只在 Python 之间下使用

json 只能序列化最基本的数据类型，而 pickle 可以序列化所有的数据类型，包括类、函数都可以序列化

https://blog.csdn.net/ITBigGod/article/details/86477083

>The [`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle) module implements binary protocols for serializing and de-serializing a Python object structure. *“Pickling”* is the process whereby a Python object hierarchy is converted into a byte stream, and *“unpickling”* is the inverse operation, whereby a byte stream (from a [binary file](https://docs.python.org/3/glossary.html#term-binary-file) or [bytes-like object](https://docs.python.org/3/glossary.html#term-bytes-like-object)) is converted back into an object hierarchy. Pickling (and unpickling) is alternatively known as “serialization”, “marshalling,” [1](https://docs.python.org/3/library/pickle.html#id7) or “flattening”; however, to avoid confusion, the terms used here are “pickling” and “unpickling”.

用 `pickle.load()` 进行序列化 serializing or pickling 和 `pickle.dump()` 进行反序列化 de-serializing or unpickling

下面的例子是 i2dl 中的 MemoryImageFolderDataset 类，用于将不大的 Dataset 放到内存中，来加快 IO 速度

```python
with open(os.path.join(self.root_path, 'cifar10.pckl'), 'rb') as f:
    save_dict = pickle.load(f)
```

## *PyYaml*

[pyyaml.org/wiki/PyYAMLDocumentation](https://pyyaml.org/wiki/PyYAMLDocumentation)

[PyYAML 使用技巧 | Reorx’s Forge](https://reorx.com/blog/python-yaml-tips-zh/)

```cmd
pip install pyyaml
pip install yaml   # py2
```

### load

PyYaml 的 load 可以构造任意 Python 对象（Pickle 协议），这意味着一次 load 可能导致任意 Python 函数被执行

**为了确保应用程序的安全性，尽量在任何情况下使用 yaml.safe_load 和 yaml.safe_dump**

* safe_load 返回一个dict
* safe_load_all函数用于加载多个yaml文档，并返回一个可选代的生成器，可以逐个获取每个文档的数据

### dump



### 遍历

## *正则*

```python 
import re
```

- 使用原始字符串（如 `r'text'`）来表示正则表达式，这样可以避免转义字符带来的困扰。关于正则表达式的书写可以看 *shell脚本和正则.md*
- 正则表达式的匹配默认是区分大小写的，可以通过设置 `re.IGNORECASE` 标志来忽略大小写

### 常用方法

* `re.match()` 尝试从字符串的**起始位置**匹配一个模式，如果不是起始位置匹配成功的话，`match()`就返回`None`

* `re.serach()` 搜索整个字符串，并返回**第一个**成功的匹配对象

* `re.findall()` 在字符串中找到正则表达式所匹配的所有子串，并返回一个列表，如果没有找到匹配的，则返回空列表

* `re.finditer()` 与 `findall()` 类似，但返回的不是直接的列表，而是一个迭代器，迭代器里的每个元素都是 match 对象

* `re.sub()` 用于替换字符串中的匹配项

* `re.compile()` 可以将正则表达式编译成一个正则表达式对象，可以用于重复使用

  ```python
  pattern = re.compile(r'\d+')
  string = '12 drummers drumming, 11 pipers piping, 10 lords a-leaping'
  found_all = pattern.findall(string)
  ```

### 捕获组

```python
import re

line = '#include "example.h"'
match = re.search('#include "(.*)"', line)
if match:
    print(match.group(1))  # 输出: example.h
```

以正则表达式 `'#include "(.*)"'` 为例：

- `#include "`: 匹配字面上的 `#include "` 字符串
- `(.*)`: 这是一个捕获组 capturing group，`.` 表示匹配除换行符 `\n` 之外的任何单个字符，`*` 表示匹配前面的字符零次或多次。所以 `.*` 可以匹配任意长度的字符串
- `"`: 匹配字面上的双引号字符

如果 `re.search` 在 `line` 中找到了匹配的内容，它会返回一个 `Match` 对象。否则，如果没有找到匹配项，则返回 `None`

`match.group(1)` 调用的是这个 `Match` 对象的 `group` 方法，它用来获取正则表达式中定义的捕获组的匹配文本

- `group(0)` 或 `group()` 将返回整个匹配的文本，即包括了 `#include "` 和闭合的双引号 `"`.
- `group(1)` 将返回第一个捕获组的内容，也就是 `(.*)` 所匹配的部分，也就是说在这个例子中，它将返回 `#include "<filename>"` 中的 `<filename>` 部分（不包含双引号）

### 原生字符串

原生字符串（raw string）是 Python 中的一种字符串字面量，它通过在字符串前面加上前缀 `r` 或 `R` 来表示。使用原生字符串时，反斜杠 `\` 不会被当作特殊字符处理，这意味着不需要对反斜杠进行转义。原生字符串常用于正则表达式和文件路径等场景，因为这些场合经常会出现需要使用到反斜杠

例如，在普通的字符串字面量中，如果想要表示一个文件路径，需要对每个反斜杠进行双重转义，因为在普通字符串中 `\` 有特殊的含义，比如 `\n` 表示换行符：

```python
# 普通字符串中的文件路径
file_path = "C:\\Users\\username\\folder"
```

但是如果使用原生字符串，则不需要对 `\` 进行转义：

```python
# 原生字符串中的文件路径
file_path = r"C:\Users\username\folder"
```

相同地，在编写正则表达式时，原生字符串可以让我们更直观地书写模式，避免了由于转义造成的混淆：

```python
import re

# 使用原生字符串定义正则表达式
pattern = re.compile(r"\bword\b")

# 在普通字符串中，需要对反斜杠进行转义
pattern = re.compile("\\bword\\b")
```

在上面的例子中，`\b` 表示单词边界，如果不使用原生字符串，那么需要写成 `\\b` 才能正确传递给 `re.compile` 函数。使用原生字符串，只需要写成 `\b`，更简洁也更容易理解

## *rich*

## *gitpython*

[GitPython Documentation — GitPython 3.1.43 documentation](https://gitpython.readthedocs.io/en/stable/)

```cmd
$ pip install gitpython
```





## *tqdm 进度条*

简介：https://zhuanlan.zhihu.com/p/163613814

Documentation：https://tqdm.github.io/docs/tqdm/

>tqdm derives from the Arabic word (taqaddum) which can mean “progress,” and is an abbreviation for “I love you so much” in Spanish (te quiero demasiado).

### 基于迭代对象运行

```python
import time
from tqdm import tqdm, trange

#trange(i)是tqdm(range(i))的一种简单写法
for i in trange(100):
    time.sleep(0.05)

for i in tqdm(range(100), desc='Processing'):
    time.sleep(0.05)
>>>100%|█████████████████████████████████████████████████████████████| 100/100 [00:05<00:00, 18.86it/s]
```

It/s 的意思是 iteration per second

### 手动进行更新

# 工程工具

## *Debugger*

```cmd
$ python3 -m pdb detector.py -f <file>
```

## *Linter*

## *Formatter*

[life4/awesome-python-code-formatters: A curated list of awesome Python code formatters (github.com)](https://github.com/life4/awesome-python-code-formatters)

### yapf

[google/yapf: A formatter for Python files (github.com)](https://github.com/google/yapf)

```cmd
$ yapf --style='{based_on_style: facebook, indent_width: 2}' -i detector.py
```

预定义风格：google, yapf, pep8 (default), facebook

类似于 `.clang-format` 文件用于 clang-format，在使用 yapf 时，可以创建一个命名为 `.style.yapf` 的文件，放在项目的根目录下或者任何父目录中，yapf 会递归向上搜索这个文件并应用它的配置

`.style.yapf` 文件中的配置格式通常如下所示：

```
[style]
based_on_style = facebook
indent_width = 2
```

当运行 yapf 命令时，它会自动查找并应用这个配置文件。如果有一个已经存在的样式配置，并且想要保存为全局配置或特定项目的配置，只需将其内容移动到 `.style.yapf` 文件即可

此外也可以指定一个自定义的配置文件路径，通过命令行中的 `--style` 选项直接传递给 yapf

```cmd
$ yapf --style=/path/to/your/configfile -i runner_cm_graph_builder.py
```

命令行中提供的任何参数将会覆盖配置文件中的设置

如果没有在命令行中指定样式，也没有提供 `.style.yapf` 文件，那么 `yapf` 将使用默认的 PEP 8 样式进行格式化

批量格式化

```cmd
$ find /path/to/directory -name '*.py' -exec yapf --style='{based_on_style: facebook, indent_width: 2}' -i '{}' +
```

