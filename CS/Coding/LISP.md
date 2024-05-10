

[为什么Lisp语言如此先进？（译文） - 阮一峰的网络日志 (ruanyifeng.com)](https://www.ruanyifeng.com/blog/2010/10/why_lisp_is_superior.html)

[初识函数式编程和Lisp之后的一点感想 - wanghui - 博客园 (cnblogs.com)](https://www.cnblogs.com/wanghui9072229/archive/2011/05/17/2048892.html)

[[转载\]LISP 语言是怎么来的–LISP 和 AI 的青梅竹马 – AI分享站 (aisharing.com)](http://www.aisharing.com/archives/648)

[LISP - 基本语法 - Lisp教程 (yiibai.com)](https://www.yiibai.com/lisp/lisp_basic_syntax.html)

# LISP (Common Lisp)

LISP, LISt Processing language 是一种编程语言，它主要用于人工智能（AI）领域的研究。LISP语言 John McCarthy 在1958年设计，并且它是继FORTRAN之后第二古老的高级编程语言

多年来，已经发展出许多LISP方言，包括Common Lisp和Scheme等。这些方言在语法和功能上存在差异，但都保留了最初LISP设计的核心概念。尽管LISP不如一些现代语言流行，但它在某些领域（如AI研究）仍然具有重要地位，并且其独特的特性和设计理念影响了许多后来的编程语言

## *运行环境*

### 安装LISP

Lisp 有很多不同的实现。比较流行的开源版本有 [SBCL](http://sbcl.org/)、[GNU Lisp](http://clisp.org/) 和 [GNU Common Lisp](https://www.gnu.org/software/gcl/)（GCL）。可以使用发行版的包管理器安装它们中的任意一个，在本文中，笔者使用的是 `clisp`（即 GNU Lisp，一种 ANSI Common Lisp 的实现）

在ubuntu上直接用包管理器来安装就行了

```cmd
$ sudo apt install clisp
```

### REPL

REPL, Read-Eval-Print-Loop 是clisp的运行时环境，和python的运行时环境很相似

## *数据类型*

LISP语言的核心在于它简单而统一的语法。它主要由S-表达式（Symbolic Expressions）和原子构成，原子包括符号（symbols）、数字等基本数据类型

### S-表达式

[S-expressions（S-表达式） (binghe.github.io)](https://binghe.github.io/pcl-cn/chap04/s-expressions.html)

[理解S表达式 | 何幻 (thzt.github.io)](https://thzt.github.io/2015/04/02/s-expression/)

````
原子 -> 数字 | 符号
S表达式 -> 原子 | (S表达式 . S表达式)
````

S表达式是用来表示数据和代码的基本结构。它们可以是原子或者由多个S-表达式组合而成的列表。所有的复合表达式都是由S表达式组合而成

- **原子 atom**：最简单的S-表达式
  - 数字 Numbers
    - 整数 Integers
    - 浮点数 Floating-point numbers
    - 有理数 Rational numbers：包括分数表示形式
    - 复数 Complex numbers
  - 字符 Characters：单个文本字符
  - 字符串 Strings：字符序列，通常被双引号包围

- 复合类型
  - 列表 List：由若干S-表达式组成且放置在圆括号内，如`(a b c)`
  - 向量 Vectors：类似于数组的数据结构，能够通过索引高效地访问元素
  - 数组 Arrays：高维的数据结构，可用于创建矩阵等结构
  - 哈希表 Hash tables：提供键-值对存储机制，允许快速访问关联数据


### 单引号

LISP计算一切，包括函数的参数和列表的成员。但有时候，我们需要采取原子或列表字面上，不希望他们求值或当作函数调用

要做到这一点，我们需要先原子或列表中带有单引号

### 运算符

下面是Lisp中的一些基本运算符，但是Lisp的强大之处在于可以用这些构建块来定义自己的运算符（实际上是函数），从而扩展语言的能力

* 数学运算符：`+ - * /`、`mod` 求模、`incf` 和 `decf` 类似于C的 `++ --`
* 比较运算符：`= /= < > <= >=`、`equal` 或 `eql` 检查两个对象是否相等（适用范围和精度有差异）
* 逻辑运算符：`and or not`

## *变量 & 常量*

### 变量

变量用来存储数据值，可以在程序运行时改变其内容。变量可以通过多种不同的方式创建和赋值

* 全局变量

  - 全局变量通常使用 `defvar` 或者 `defparameter` 关键字定义

  - `defvar` 仅在变量未被定义时初始化其值，而 `defparameter` 总是重新初始化变量
  - 注意：Lisp 社区中有一个约定，全局变量名通常以 `*` 符号包围

  ```lisp
  (defvar *global-var* "Initial value")  ; 创建一个全局变量并初始化
  (defparameter *another-global* 123)    ; 创建另一个全局变量并初始化
  ```

* 局部变量

  - 局部变量通常使用 `let` 和 `let*` 构造创建

  - `let` 用于创建新的作用域，并在这个作用域内定义变量

  - `let*` 类似于 `let`，但是允许后续的变量定义依赖于前面的变量定义

  ```lisp
  (let ((local-var 10)
        (another-local "Hello"))
    ;; 在这里 local-var 和 another-local 是可用的
    )
  ```

### 常量

在LISP中，常量变量在程序执行期间，从来没有改变它们的值。常量使用 `defconstant` 声明

```lisp
(defconstant +pi+ 3.14159) ; 创建一个常量表示圆周率
```

## *分支控制*

### 条件

* if

  ```lisp
  (if condition then-part else-part)
  
  ; 一个例子
  (if (> x 0)
      (print "x is positive")
      (print "x is non-positive")) 
  ```

* cond 是更复杂的条件控制结构，可以支持多个分支。类似于switch

  ```lisp
  (cond
    (test1 result1)
    (test2 result2)
    ...
    (t default-result))
  
  ; 一个例子
  (cond
    ((> x 10) (print "x is greater than 10"))
    ((< x 5)  (print "x is less than 5"))
    (t        (print "x is between 5 and 10 inclusive")))
  ```

  每个分支都有一个测试条件（`test1`, `test2`, ...）和对应的结果（`result1`, `result2`, ...）。如果一个测试条件为真，它对应的结果就会被执行。`t` 对应于 `default`，它在所有其他条件都不满足时执行

* when 和 unless 是带有内建条件的特殊情况。`when` 执行内部的表达式，仅当 `condition` 为真时；而 `unless` 则相反，仅当 `condition` 为假时才执行内部的表达式

  ```lisp
  (when condition
    form1
    form2
    ...)
  
  (unless condition
    form1
    form2
    ...)
  
  ; 一个例子
  (when (> x 10)
    (print "x is greater than 10"))
  
  (unless (<= x 10)
    (print "x is greater than 10"))
  ```

### 循环

* loop：循环loop结构是迭代通过LISP提供的最简单的形式。在其最简单的形式，它可以重复执行某些语句(次)，直到找到一个return语句
* loop for：loop结构可以实现一个for循环迭代一样作为最常见于其他语言
* do：do 结构也可用于使用LISP进行迭代。它提供了迭代的一种结构形式
* dotimes：dotimes构造允许循环一段固定的迭代次数
* dolist：dolist来构造允许迭代通过列表的每个元素

## *函数*

### 列表和函数调用

LISP中的函数调用也表示为S-表达式。一个列表如果作为代码执行，则其第一个元素通常是函数或操作符，后续元素是该函数的参数

例如，计算两数之和的表达式写作：`(＋ 3 4)`

```lisp
[1]> (+ 3 4)
7
```

这里的 `＋` 是函数名（在LISP中是一个符号），`3` 和 `4` 是它的参数。当这个S-表达式被求值时，会执行加法运算并返回结果 `7`

### 定义函数

除了普通的函数调用，LISP也有几个特殊形式的构造，如用于条件判断的`if`，以及用于定义新函数的`defun`

```lisp
(defun my-function (param1 param2)
  ;; 在这里 param1 和 param2 是函数参数变量
  )
```

例如，定义一个加法函数的方式如下所示：

```lisp
(defun add (x y)
  (+ x y))
```

这里，`defun` 是一个特殊形式，用于定义函数 `add`，该函数接受两个参数 `x` 和 `y` 并返回它们的和。

### 谓词

谓词是函数，测试其参数对一些特定的条件和返回nil，如果条件为假，或某些非nil值条件为true

### 宏（Macros）

LISP的宏系统允许用户扩展语言的语法，创建自己的特殊形式。宏看起来像函数调用，但是它们在编译时间进行代码转换，而不是在运行时求值

使用 `defmacro` 来定义宏

```lisp
(defmacro macro-name (parameter-list)
 "Optional documentation string."
 body-form)
```

### 示例程序

下面是一个简单的LISP程序示例，计算阶乘的一个函数：

```lisp
(defun factorial (n)
  (if (= n 0)
      1
      (* n (factorial (- n 1)))))
```

在这个函数中，如果 `n` 等于 `0`，就返回 `1`；否则，返回 `n` 乘以 `n-1` 的阶乘。

整体上，LISP的语法非常简洁，几乎所有的结构都可以表示为嵌套的S-表达式。这种设计使得LISP代码具有高度的灵活性，能够轻松地表示复杂的数据结构和算法

## *类和结构体*

### 结构体

### 类

## *实现一个LISP的语法解析器*

[736. Lisp 语法解析 - 力扣（LeetCode）](https://leetcode.cn/problems/parse-lisp-expression/description/)

# Scheme

Scheme是LISP编程语言家族的一个成员，它是由Guy L. Steele和Gerald Jay Sussman设计并首次介绍于1975年。Scheme的设计哲学强调极简主义，目的在于创建一个干净而高效的LISP方言，特别是在实现尾递归和闭包方面

与其他LISP方言类似，Scheme的语法非常简单，同样几乎全部建立在S-表达式之上。Scheme在教育界特别流行，经常被用来教授计算机科学和程序设计入门课程。此外，它也因其优雅的设计和强大的语言特性，被一些专业程序员用来进行软件原型设计和实验性项目开发

## *方言*

在编程语言的领域，**方言 dialect**指的是基于某个主要编程语言派生出来的具有相似性但又有一定区别的编程语言。方言通常保留了原始语言的核心特性和理念，同时引入了新的特性或者改变了部分现有特性以满足特定需求。

方言的产生可以由多种原因推动，包括但不限于：

1. **平台兼容性**: 某些方言可能针对特定操作系统或硬件平台进行优化。
2. **社区驱动**: 开发者社区可能会根据他们的特定需求对语言进行调整。
3. **实验特性**: 方言也可能是语言设计者引入新特性的试验场，用来探索语言设计的未来方向。
4. **教育目的**: 为了教学的简便性，可能会创建简化的版本，去除一些复杂的特性。
5. **历史遗留**: 随着时间的推移，语言的不同版本可能因标准化的缺乏而形成方言。

举例来说，在LISP编程语言家族中，Common Lisp和Scheme都是LISP语言的方言，它们继承了LISP的核心概念，如使用S-表达式表示代码和数据，以及强大的宏系统。然而，它们在语言特性、语法和实现上有所区别：Common Lisp较为复杂，拥有丰富的内置功能，而Scheme则注重简洁性，提供小巧且灵活的核心。

## *运行环境*

## *基本语法*

### 函数调用

Scheme中函数调用是通过将函数名和参数放置在一个列表中实现的。函数调用的通用格式如下：

```scheme
(function-name arg1 arg2 ... argN)
```

例如，计算两数之和的函数调用写作：`(＋ 3 4)`，会返回结果 `7`。

### 定义变量和函数

- **定义变量**: 使用 `define` 关键词可以创建新的变量并赋值。

  ```
  scheme复制代码(define x 10) ; 定义变量x并赋值为10
  ```

- **定义函数**: `define` 也用来定义函数，通常与 `lambda` 表达式一起使用。

  ```
  scheme复制代码(define (square x)
    (* x x))
  ```

### 特殊形式

Scheme中有一系列特殊形式，用于控制结构、变量绑定等功能。

- **条件判断**: 使用 `if` 和 `cond` 进行条件分支。

  ```
  scheme复制代码(if (> x 5)
      'greater
      'not-greater)
  
  (cond ((> x 5) 'greater)
        ((< x 5) 'lesser)
        (else 'equal))
  ```

- **Lambda表达式**: 创建匿名函数。

  ```
  scheme复制代码(lambda (x) (* x x))
  ```

- **Let绑定**: 在局部作用域内绑定变量。

  ```
  scheme复制代码(let ((x 2)
        (y 3))
    (+ x y))
  ```

### 宏定义

Scheme提供了强大的宏系统，允许用户定义自己的语法结构。

```
scheme复制代码(define-syntax my-when
  (syntax-rules ()
    ((my-when test expr ...)
     `(if ,test
           (begin
             ,@expr)))))
```

### 注释

- 单行注释使用分号 `;` 开头。
- 多行注释使用 `#|` 开始，`|#` 结束。