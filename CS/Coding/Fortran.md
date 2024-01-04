## *Fortran介绍*

Fortran（Formula Translation）是一种编程语言，最初是为科学和工程计算而设计的。它是最早的高级编程语言之一，于1957年由IBM开发。Fortran的设计目标是提供一种方便、高效的编程方式，使科学家和工程师能够更容易地进行数学和科学计算。

以下是一些关于Fortran的主要特点和历史背景：

1. **数学和科学计算：** Fortran主要用于数学和科学计算，因此它在处理复杂的数值计算和大规模数据集方面表现出色。
2. **数组操作：** Fortran具有强大的数组处理能力，这使得在科学和工程领域中处理矩阵、向量和多维数组变得非常方便。
3. **历史：** Fortran的第一个版本是在1957年发布的，之后经历了多个版本的更新。Fortran 77是一个非常流行的版本，它在20世纪70年代和80年代广泛使用。后来，Fortran 90、95、2003、2008等版本陆续发布，引入了新的功能和语法。
4. **易读性：** Fortran代码通常以列格式书写，这是其与其他编程语言不同之处。Fortran的代码结构使得数学公式和方程式能够以更自然、直观的方式呈现。
5. **并行计算：** Fortran具有一些特性，使得它适合并行计算，这在当今高性能计算环境中非常重要。
6. **科学与工程应用：** Fortran在气象学、物理学、工程学、地球科学等领域广泛应用。很多现有的科学计算代码库和应用程序仍然使用Fortran编写。

Fortran的语法主要关注科学计算的需求，因此在数学表达、数组操作和高性能计算方面有着很强的优势

## *程序结构*

Fortran程序通常由一系列的子程序组成，其中最主要的是程序单元（Program Unit）。一个程序单元可以是主程序（Main Program）或子程序（Subroutine）。主程序是整个程序的入口，而子程序是执行具体任务的独立单元

用 PROGRAM 定义主程序的开始和结束

```fortran
PROGRAM MainProgram
   ! 主程序语句
END PROGRAM MainProgram
```

### 注释

注释可以使用感叹号（`!`）进行，可以是行内注释或者单行注释

```fortran
! 这是一个注释
a = b ! 这也是一个注释
```

### 子程序 & 模块 & 函数

* Fortran程序可以包含子程序，通过使用子程序可以更好地组织代码和实现模块化设计。不返回数值结果

  ```fortran
  SUBROUTINE MySubroutine(arg1, arg2)
     ! 子程序的语句
  END SUBROUTINE MySubroutine
  ```

* 函数：可以返回一个值

  ```fortran
  REAL FUNCTION MyFunction(arg1, arg2)
     ! 函数的语句
  END FUNCTION MyFunction
  ```

* 模块：模块可以包含数据（变量）、子程序、函数等，这些在模块中的元素可以在其他程序单元中使用

  ```fortran
  MODULE MyModule
     INTEGER :: variable1
     CONTAINS
     SUBROUTINE MySubroutine(arg1, arg2)
        ! 子程序的语句
     END SUBROUTINE MySubroutine
  END MODULE MyModule
  ```

## *变量*

### 变量声明

变量在使用之前需要声明，声明包括变量名、数据类型以及可选的初始化值。

```fortran
INTEGER :: a, b = 5
REAL    :: x, y
```

`::` 是声明符号，后面列出了需要声明的变量名称，每个变量后面的 `INTEGER` 或 `REAL` 是数据类型。也可以在声明时为变量赋初值

在早期的Fortran版本中，允许隐式声明。如果变量名称以字母 `i`、`j`、`k`、`l`、`m` 或 `n` 开头，则被隐式声明为整数类型。其他字母开头的变量被隐式声明为实数类型。然而，在现代的Fortran标准中，推荐使用显式声明，以提高代码的可读性和可维护性

### 变量类型

* 整数类型INTEGER：用于存储整数值，可以是正整数、负整数或零

   ```fortran
   INTEGER :: a, b = 5
   ```

* 实数类型 REAL：用于存储浮点数（带有小数点的数值），包括单精度和双精度

   ```fortran
   REAL :: x, y = 3.14
   ```

* 复数类型 COMPLEX： 用于存储复数，包括实部和虚部

   ```fortran
   COMPLEX :: z, w = (2.0, 3.0)
   ```

* 逻辑类型 LOGICAL：用于存储逻辑值，即真（.TRUE.）或假（.FALSE.）

   ```fortran
   LOGICAL :: flag, status = .TRUE.
   ```

* 字符类型 CHARACTER：用于存储字符串，需要指定字符串的长度

   ```fortran
   CHARACTER(10) :: name
   ```

* 数组类型：Fortran支持多维数组，可以是整数数组、实数数组、复数数组等

   ```fortran
   INTEGER, DIMENSION(3, 3) :: matrix
   ```

* 派生类型 DERIVED：允许用户定义自己的数据类型，包括多个成员变量，形成一个结构

   ```fortran
   TYPE MyType
      INTEGER :: member1
      REAL    :: member2
   END TYPE MyType
   
   TYPE(MyType) :: obj
   ```

### 数组

Fortran对数组的支持非常强大，数组的声明和使用相对简单

```fortran
REAL, DIMENSION(3, 3) :: matrix
matrix(1, 1) = 1.0
```

Fortran也支持动态数组

```fortran
REAL, ALLOCATABLE :: dynamic_array(:)
```

## *控制流*

### 条件控制

```fortran
IF (condition) THEN
   ! 条件为真时执行的语句
ELSE
   ! 条件为假时执行的语句
END IF
```

```fortran
SELECT CASE (variable)
   CASE (value1)
      ! 执行语句
   CASE (value2)
      ! 执行语句
   CASE DEFAULT
      ! 默认执行语句
END SELECT
```

### 循环

```fortran
DO index_variable = start, stop [, step]
   ! 循环体
END DO
```

`index_variable` 是循环索引变量，其值从 `start` 开始逐步增加，直到达到或超过 `stop`。可选的 `step` 参数表示每次迭代的步长，如果在DO循环中未指定step，系统将默认步长为1





```fortran
DO J = 1, M, T
  DO I = 1, N, T
  	DO ii = I, min(I+T-1, N)
      DO jj = J, min(J+T-1, M)
        A(ii) = A(ii) + B(jj)
      END DO
    END DO
  END DO
END DO
```



