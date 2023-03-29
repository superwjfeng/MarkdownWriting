# 信息的表示和处理

## *信息存储*

* 整形表示方式

  * 原码

  * 反码

  * 补码

  * 转换关系

    <img src="/Users/wjfeng/Desktop/课程大纲/CS/C/机器码转换关系.jpg" width="50%">

* 进制转换

  <img src="/Users/wjfeng/Desktop/课程大纲/CS/C/进制转换关系.jpg" width="50%">

* 大小端介绍

  <img src="/Users/wjfeng/Desktop/课程大纲/CS/C/大小端.png" width="45%">

  * 什么是大端和小端

    * 大端 Big endian：**低地址高字节**，即**数据的高字节保存在内存的低地址中，而数据的低字节保存在内存的高地址中**，比较符合阅读习惯
    * 小端 Little endian：**低地址低字节**

  * 判断内存是以大端存储还是小端存储

    * 利用指针

      ```c
      int i = 1; // 01 00 00 00
      if (1 == *(char*)&i) // 将int*强制转换成char*后对该指针解引用，char*只能得到1字节的内存
          printf("Little endian\n");
      else
          printf("Big endian\n");
      ```

    * 利用联合体

      ```c
      int CheckSys()
      {
          union Un
          {
              char c;
              int i;
          }u;
      
          u.i = 1;
          return u.c; // 两者共用地址，若是小端则返回1，否则返回0
      }
      ```

## *整数表示*

### 整型数据类型

C语言标准定义了每种数据类型必须能够表示的**最小的取值范围**

### 无符号数的编码

设一个有 $w$ 位的整数数据类型，有位向量 $\vec{x}=\left[x_{w-1},x_{w-2},\cdots,x_{0}\right]$，其中每个位 $x_i$ 的取值为0或1。用一个函数 $B2U_w$ Binary to Unsigned 来表示从二进制到无符号数
$$
B2U_w\left(\vec{x}\right)\triangleq\sum\limits_{i=0}^{w-1}{x_i2^i}
$$
一个 $w$ 为的二进制位向量 $\vec{x}$ 可以表示的最大无符号数为 $UMax_w$，它的二进制为 $\left[11\cdots1\right]_w$
$$
UMax_w\triangleq\sum\limits_{i=0}^{w-1}{2^i}=2^w-1
$$
无符号数编码具有唯一性，因为 $B2U_w$ 是一个双射 bijectio，即可以进行反向操作从无符号数唯一映射到二进制，用函数 $U2B_w$ 来表示

### 补码 two's-complement

C语言标准并没有要求要用补码形式来表示有符号整数，但是几乎所有的机器都是这么做的

为了表示负数，引入了原码 Sign-Magnitude、反码 Ones' complement 和补码表示。因为原码和反码对0的表示有两种和其他原因，补码是最好的选择

补码的最高有效位 $x_{w-1}$ 称为符号位 sign bit，被解释为负权 negative weight，它的权重为 $-2^{w-1}$。用函数 $B2T_w$ Binary to Two's-complement 定义
$$
B2T_w\left(\vec{x}\right)\triangleq-x_{w-1}2^{w-1}+\sum\limits_{i=0}^{w-2}{x_i2^i}
$$
补码能表示的最小值是位向量 $\left[10\cdots0\right]_w$，最小整数值是 $TMin_w=-2^{w-1}$；最大值的位向量是 $\left[01\cdots1\right]_w$，最大整数值为 $TMax_w=\sum\limits_{i=0}^{w-2}{2^{w-1}-1}$ 

$B2T_w$ 是一个双射，是从 $TMin_w$ 到 $TMax_w$ 之间的映射，即 $B2T_w:\left\{0,1\right\}^w\rightarrow\left\{TMin_w,\cdots,TMax_w\right\}$。它的反函数为 $T2B_w$

### 有符号数和无符号数之间的转换

强制类型转换的结果保持位值不变，只是改变了解释这些位的方式
$$
T2U_w(x)\triangleq B2U_w\left(T2B_w(x)\right),\ x\in\left[TMin_w,TMax_w\right]\\U2T_w(x)\triangleq B2T_w\left(U2B_w(x)\right),\ x\in\left[0,UMax_w\right]
$$
给定位模式的补码与无符号数之间的关系可以表示为函数 $T2U_w$ 的一个属性
$$
T2U_w(x)=\left\{\begin{array}{cc}x+2^w,&x<0\\x,&x\geq0\end{array}\right.
$$


## *整数运算*

## *浮点数*

# 程序的机器级表示

# 处理器体系结构

# 优化程序性能

# 存储器层次结构