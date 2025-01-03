## *Intro*

Lua有着相对简单的C语言API而很容易嵌入应用中，例如游戏引擎、服务器端扩展等。很多应用程序使用Lua作为自己的嵌入式脚本语言，以此来实现可配置性、可扩展性

### Lua vs. Shell

* 语法和语言结构

   * Lua 是一种通用的嵌入式脚本语言，具有简单、清晰的语法。它使用 `end` 来标记代码块，使用 `--` 进行单行注释，而且支持面向对象编程

     ```lua
     for i = 1, 5 do
       print(i)
     end
     ```

   * Shell 脚本通常基于命令行，语法更接近于命令行命令。Shell 使用 `fi` 结束 `if` 语句，`done` 结束循环，使用 `#` 进行注释。

     ```shell
     for i in {1..5}; do
       echo $i
     done
     ```

* 设计目标

   * Lua 被设计为一种通用的嵌入式脚本语言，旨在提供简单、灵活、高效的脚本执行环境。它经常用于嵌入到其他应用程序中，例如游戏引擎、服务器端扩展等
   * Shell 脚本主要用于在命令行环境中执行系统命令和自动化任务。它通常用于处理文件、执行系统命令、文本处理等

* 数据类型和结构

   * Lua 是一种脚本语言，具有多种数据类型，例如数字、字符串、表等。它支持面向对象编程，并提供了强大的表（table）数据结构用于实现数组、字典等数据结构
   * Shell 脚本的数据类型相对简单，通常包括字符串、数字等。数组和关联数组的支持相对较弱，而且它们通常通过字符串来表示

## *基础语法*

### 注释

* 使用 `--` 进行单行注释

* 块注释/多行注释

  ```lua
  --[[
  这是一个多行注释
  可以跨越多行
  --]]
  local variable = 42
  ```

### 变量声明

当使用 `local` 关键字声明一个变量时，该变量的作用域被限制在当前块（通常是一个函数或一个控制结构块）内

```lua
-- 全局变量
globalVariable = 10

-- 声明局部变量
local localVariable = 5

-- 函数定义
function myFunction()
  -- 这里的局部变量仅在函数内可见
  local innerVariable = 20
  print(globalVariable)  -- 可以访问全局变量
  print(localVariable)   -- 可以访问函数外声明的局部变量
  print(innerVariable)   -- 可以访问函数内声明的局部变量
end

myFunction()

-- 在这里访问 innerVariable 会导致错误，因为它是在函数内部声明的局部变量，作用域仅限于该函数。
-- print(innerVariable)  -- 这一行会产生错
```

### 数据类型

Lua 是一种动态类型语言，这意味着变量的数据类型是根据其当前值自动确定的，所以不需要声明数据类型

* nil空值：表示无效或未赋值的变量，默认值是 nil。
* boolean：true 和 false
* number：表示实数（浮点数）或整数。Lua 中只有一种数字类型，可以表示整数或浮点数
* string：表示文本数据。字符串可以用单引号或双引号括起来。例如，`"Hello, Lua!"` 或者 `'Lua is awesome!'`
* table 表：是 Lua 中的复合数据类型，可以看作是一种关联数组或字典。表用来存储键值对
* function 函数：是一等公民，可以赋值给变量，作为参数传递，也可以作为返回值。函数可以通过 `function` 关键字定义
* userdata 用户数据：是一种允许用户在 Lua 中创建自定义数据类型的机制。通常用于与 C 语言等外部语言进行交互
* thread：是 Lua 中的协程实现，用于支持协同式多任务处理

## *控制流*

使用 `end` 来标记代码块

### 条件控制

### 循环

## *函数*

## *模块*

在Lua中，`require` 是一个用于加载和执行其他Lua模块的函数。它是Lua语言中用于模块化编程的一种重要机制