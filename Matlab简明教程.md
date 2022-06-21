# Matlab 简明教程 from Matlab Documentation

## Matlab快速入门

### 工作区变量

* 使用whos来查看工作区变量
* 使用save以.mat扩展名将工作区变量保存在MAT压缩文件中
* 使用load将MAT文件中的数据还原到工作区

### 数组类型

* 多维数组
* Cell元胞数组
* 文本和字符
  * 使用双引号创建字符序列

    ```matlab
    t = "Hello, World!";
    ```

  * 使用+将字符串连接起来
* 结构体


### 控制流

* 需要加end结尾匹配
* if else switch
* for while continue break
* 程序终止：return，matlab函数不是通过return来返回的，而是直接通过赋值给返回值进行返回，return用来返回控制权

## 语言基础知识

### 矩阵和数组

* 所有Matlab变量都是多维数组，与数据类型无关。

#### 创建、串联和扩展矩阵

* 创建矩阵：和C语言不同，创建矩阵/数组时不需要指定大小，除非是用函数创建

    ```matlab
    A = [1, 2, 3] % 3*1 矩阵
    B = [1 2 3] % 1*3 矩阵
    ```

  * zeros
  * ones
  * eye
  * rand 均匀分布的随机元素
  * randn 正态分布的随机元素

* 串联：用[]连接数组以形成更大数组

    ```matlab
    A_con = [A; A]
    ```

* 用 : 生成数值序列

#### 重构和重新排列数组

* reshape用于重构数组的大小和形状
* 转置和翻转
  * ' transpose
  * flip
* 平移和旋转 
  * circleshift
  * rot90
* 排序
  * sort(A, 2, 'descend')按降序对A的每一行进行排序；若要对A中的所有元素进行排序然后输出一个列向量，则要用sort(A(:), 'descend')
  * sortrows

#### 索引

* 数据索引使用()
* 单独的 : 指定该维中的所有元素
* : 还允许使用格式start:step:end创建等间距向量值
* 使用[\]来删除行和列，若使用[]来删除单个元素会产生错误

### 数据类型

#### 数值类型

* 整数类：MATLAB默认情况下以双精度浮点形 double存储数值数据。若要用整数形存储数据，需要用int进行转换。若其带有小数部分，将会被舍入，也可使用round fix floor ceil函数切换舍入

#### 函数句柄 Function Handle

函数句柄是一种存储指向函数的关联关系的Matlab数据类型，类似于C语言中的函数指针

* 创建函数句柄：可以为已命名函数和匿名函数创建函数句柄，使用@来创建函数句柄

    ```matlab
    function y = computeSquare(x)
    y = x.^2;
    end

    f = @computeSquare;
    a = 4;
    b = f(a);
    ```

  * 函数句柄数组

* 将一个函数传递到另一个函数
使用函数句柄作为其他函数（称为复合函数）的输入参数，这些函数基于某个范围内的值计算数学表达式
  * integral

    ```matlab
    a = 0, b= 5;
    q1 = integral(@log, a, b);
    ```

  * quad2d
  * fzero
  * fminbnd
* 参数化函数
* 使用函数句柄调用局部函数
* 比较函数句柄

### 运算符和基本运算

#### 算术运算

* 基本算数
* 摸除法和舍入
  * ceil 向上舍入
  * fix 向零方向舍入
  * floor 向下舍入
  * round 
* 自定义二元函数

#### 逻辑运算

* 具有短路功能的逻辑运算 && ||
* & ~ | xor
* all any
* true false
* find
* islogical logical

### 循环及条件语句

## 数学

### 初等数学

#### 算术运算

* 数组与矩阵运算
* 

### 线性代数

### 随机数生成

#### 创建随机数数组

MATLAB使用算法来生成伪随机数和伪独立数。这些数再数学意义上并非严格随机和独立的，但他们能够通过各种随机和独立统计测试

* 随机数函数
  * rand(dim) 返回介于(0, 1)的随机数数组 rand返回一个随机数
  * randi 返回离散均匀分布中的double整数
  * randn 返回标准正态分布中的float实数数组
  * randperm 没有重复值的double数组
* 随机数生成器rng
  * 'twister' 梅森旋转为MATLAB默认流使用的算法
  * 'simdTwister' 面向SIMD的快速梅森旋转算法
* 随机数默认情况下为double

#### 特定范围内的随机数

#### 随机整数

#### 具有特定$\mu$和$\sigma^2$的float数组

```matlab
rng(0, 'twister') %初始化生成器
mu = 500;
sigma = 5;
y = sigma.*randn(1000, 1)+ mu;
```

### 插值

### 优化

### 数值积分和微分方程

### 傅里叶分析和滤波

#### 傅里叶变换

#### 基本频谱分析

#### 二维傅里叶变换

#### 使用卷积对数据进行平滑处理

* conv 
* conv2
  * full
  * same：输出和A尺寸相同的卷积结果，以在边界进行了零填充
  * valid
* convn
* deconv

#### 滤波数据

### 稀疏矩阵

### 图和网路算法

### 计算几何学

## 数据导入和分析

## 图形

### 二维图和三维图

* 二维图

```matlab
x = linspace(0, 2*pi);
y = sin(x);
plot(x, y)
```

* 三维图
  * plot3(x, y, z)画3D曲线图
  * surf(x, y, z)画3D曲面图

    ```matlab
    x = linspace(-2,2,20);
    y = x';
    z = x .* exp(-x.^2 - y.^2);
    surf(x,y,z)；
    ```

  * mesh(x, y, z)画3D曲面网格图
* 多个三维图排列
  * subplot(m, n, p)将当前图窗分为m*n网格，并在p处创建图片
  * tiledlayout(R2019b)

```matlab
subplot(2,1,1);
x = linspace(0,10);
y1 = sin(x);
plot(x,y1)

subplot(2,1,2); 
y2 = sin(5*x);
plot(x,y2)
```

### 格式和注释

#### 范围、刻度和网格

#### 多个绘图

#### 大小和纵横比

### 图像

为了节省存储空间，MATLAB使用uint8存储图像，但是当进行计算时，MATLAB使用double进行计算和存储，因此在导入图像后需要将图像转换为double后再使用
im2double im2unit8 im2unit16

#### 图像类型

* 索引图像
* 灰度图像
* RGB图像

## 脚本

## 函数

### 函数创建

#### 在文件中创建函数

将多个命令存储在一个可以接受输入和返回输出的程序文件中

#### 函数类型

* 匿名函数：匿名函数是不存储在程序文件中，但与数据类型是function_handle的变量相关的函数（类似于C语言中的宏）。使用匿名函数的好处是不必为仅需要简短定义的函数单独编辑和维护函数文件
* 局部函数：一个函数文件中可以包含多个函数。第一个函数称为主函数，此函数对其他文件中的函数可见，也可以直接从命令行调用。而其他的函数称为局部函数，局部函数仅对同一函数文件中的其他函数可见
* 嵌套函数
* 私有函数

#### 函数优先顺序

当前作用域的多个函数具有相同名称时如何确定要调用的函数，当前作用域包括当前文件、相对于当前运行的函数的可选私有子文件夹、当前文件夹以及MATLAB路径。

变量->名称与显示导入的名称匹配的函数或类->

### 参数定义

#### 参数值

* 解析函数输入的方法
* 函数参数验证
* 通过validateattributes检查函数输入
* 解析函数输入：定义必须和可选的输入、指定可选输入的默认值以及使用输入解析器inputParser验证自定义函数的所有函数
  * 定义函数：在名为printPhoto.m文件中创建函数，该函数具有必须输入的文件名，可选输入为抛光、颜色空间、宽度和高度

    ```matlab
    function printPhoto(filename, varargin)
    ```

  * 创建一个inpuitParser对象

    ```matlab
    p = inputParser;
    ```

  * 将输入添加到方案中

    ```matlab
    defaultFinish = 'glossy';
    validFinished = {'glossy', 'matte'};
    checkFinish = @(x) any(validatestring(x, validFinished));

    defaultColor = 'RGB';
    validColors = {'RGB', 'CMYK'};
    checkColor = @(x) any(validatestring(x, validColors));

    defaltWidth = 6;
    defaltWidth = 4;

    addRequired(p, 'filename', @ischar);
    addOptional(p, 'finish', defaultFinish, checkFinish);
    addOptional(p, 'color', defaultColor, checkColor);
    addParameter(p, 'width', defaultWidth, @isnumeric);
    addParameter(p, 'height', defaultHeight, @isnumeric);      
    ```

    * 验证函数必须接收单个输入参数，并返回true false，输入解析器返回错误则函数停职处理
    * 可使用现有MATLAB函数的句柄，如ischar isnumeric
    * 可创建匿名函数并输入其句柄
    * 使用自定义函数，往往是局部函数

  * 设置属性以调整解析（可选）

    ```matlab
    p.KeepUnmatched = true;
    ```

  * 解析输入

    ```matlab
    parse(p, filename, varargin{:});
    ```

  * 在函数中使用输入
  * 调用函数

#### 参数数量

* 支持可变数量的输入：使用varargin定义接收可变数量的输入参数的函数。varargin参数是包含函数输入的Cell数组，其中每个输入都位于它自己的Cell中
* 支持可变数量的输出：使用varargout定义返回可变数量的输出参数的函数
* 查找函数函数的数量：使用nargin和nargout确定函数收到的输入或输出参数的数量

#### 直通输入

### 作用域变量和生成名称

### 错误的处理方式

## 实时脚本和函数

## 类

## 文件和文件夹

### 搜索路径

* addpath 向搜索路径中添加文件夹
* rmpath 从搜索路径中删除文件夹

### 文件操作

### 文件压缩

### 文件名的构造

## 编程实用工具


* 匿名函数


### 调用函数




改变现实的变量类型format short, short e, short g, long, long e, long g, bank, rat, hex
数组



