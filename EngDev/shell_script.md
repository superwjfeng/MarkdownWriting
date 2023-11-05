# Shell脚本与正则表达式

## *helloworld*

```sh
#! /bin/bash
echo "hello, world"
```

脚本以 `#!/bin/bash` 开头来制定解析器 parser

### 执行方式

* 调用子bash或子sh来执行 `bash helloworld.sh`，嵌套执行可能会影响环境变量
* 用绝对路径或相对路径来执行脚本文件
* 用 `source`（csh实现）或 `.`（bash实现） 命令来执行脚本文件。source调用并加载新的配置到子bash

## *变量*

### 分类

* [环境变量](#环境变量)和自定义变量
* 全局和局部
  * 全局对所有bash有效，局部只对当前bash有效
  * 用户自定义的默认都是局部变量，可以通过 `export` 来导出为全局变量。但是特别的是对于子shell中更改了全局变量，不会对父shell造成影响

使用一个定义过的变量，只要在变量名前面加 `$` 符号即可

### 变量定义的规则

* 变量名称可以由字母、数字和下划线组成，但是不能以数字开头，环境变量名建议大写

* 等号两侧不能有空格

* 变量的值如果有空格，需要使用双引号或单引号括起来

* 在bash中，**变量默认类型都是字符串类型**，无法直接进行数值运算

  * 要进行数值运算要用运算符 operator 表达

    ```shell
    a=$((1+5)) # (()) 里面可以用<=这种数学表达式
    a=$[5+9]
    ```

  * 或者用比较麻烦的 `expr` 数值计算命令： `expr 1 + 2` 的时候中间要有空格，而且 `*` 要转义，即 `\*`

  * 命令替换：将一个命令运行的结果赋值给变量

    ```shell
    a=$(expr 5 \* 3) # or
    a=`expor 5 \* 3`
    ```


### 变量定义基本句法

* 定义变量：变量名=变量值，注意不能有空格，否则会被解释为两条语句
* 撤销变量：unset变量名
* 声明静态/只读变量：readonly变量 `readonly b=5`

### 特殊变量

* `$n`：n是数字，`$0` 代表该脚本名称， `$1-$9` 代表第1-9个参数，10 以上的参数要用花括号括起来 `${10}`

  ```shell
  #! /bin/bash
  echo "Hello, "
  echo $0
  echo $1
  echo $2
  ```

* `$#`：参数个数统计，获取所有输入参数的数量。常用于循环，判断参数的个数是否正确以及增强脚本的健壮性

* `$?`：最后一次执行的命令的返回状态，若为0则上次命令正确指令，非0则执行不正确

* `$*`：代表命令行中所有的参数，`$*` 把所有的参数看成一个整体

* `$@`：也代表命令中所有的参数，但是 `$@` 把每个参数区分对待，即形成一个集合或者数字

## *条件判断*

### 基本语法

* `test condition`

* `[ condition ]` 注意condition前后的空格不可少

  ``` shell
  a=hello
  [ $a = hello ] # 注意condition的空格也不能少，$a=hello 会被理解为一个整体，一个整体是非空的，从而输出0
  [ ] #echo $? 空输出1
  ```

和高级语言相反，shell script用0表示真，非0表示假，因为用的是返回状态作为判断条件

### 常用判断条件

* 两个整数之间比较：-eq 等于 -ne 不等于 -lt 小于 -le 小于等于 -gt 大于 -ge 大于等于

  ```shell
  [ 2 -lt 8 ] #echo $? 输出0 
  ```

* 两个字符串之间的比较：用等号=判断相当，用 != 判断不等

* 按照文件权限进行判断：-r -w -x 是否有读、写、执行的权限

* 按照文件类型进行判断：-e -f -d 文件是否存在 existence、存在并且是一个常规文件 file 、存并且是一个目录 directory

* 多条件判断：`&&` 表示前一条指令执行成功时，才执行后一条命令；`||` 表示上一条命令执行失败后，才执行下一条命令

  ```shell
  a=15
  [ $a -lt 20 ] && echo "$a < 20" || echo "$a >= 20" # 若判断成功则执行中间的，判断失败则执行后面的
  # 和三目运算符一样
  ```

## *流程控制*

### if语句

```shell
if [ condition ]
then
	程序
elif [ condition ]
then
	程序
else
	程序
fi
```

一个小优化保证判断条件不为空，""是进行字符串拼接（用单引号''就不会翻译$1了）

```shell
if [ "$1"x  = "zhang3"x ]; then echo "welcome, zhang3"; fi
```

若是多个判断条件有两种表示方法

```shell
if [ $a -gt 18 ] && [ $a -lt 35 ]; then echo OK; fi
if [ $a -gt 18 -a $a -lt 35 ]; then echo OK; fi # -a -r 表示逻辑与、或
```

### case语句

case就是switch语句

```shell
case $变量名 in
"value 1")
	# 程序1
;; # break
"value 2")
	# 程序2
;;
*)
	# 这里是default
;;
esac
```

## *循环*

### for循环

for循环有两种书写方式

```shell
# 书写方式1
for (( initialization;control;increment )) #(())里的循环变量不用在外面定义
do
	#程序
done

# 书写方式2
for 变量 in value 1 value 2 value 3 # 也可以表示成 for 变量 in (value 1 ... value n)
do
	#程序
done
```

一个例子

```shell
for (( i=1; i <= $1; i++ )) # (()) 里面可以用<=这种数学表达式
do
	sum=$[ $sum + $i ]
done
```

### while循环

```shell
while [ condition ] #[]里的循环变量要在外面定义
do
	程序
done
```

### read读取控制台输入

`read -p -t` -p指定读取值时的提示符；-t指定读取时等待的时间，若不加-t则一直等待

## *函数*

### 系统函数

* `basename [string/pathname][suffix]` 取路径里的文件名称，会去掉所有前缀（包括最后一个/），默认保留suffix，若指定了suffix，就会把文件的suffix后缀去掉
* `dirname`：截取绝对路径名称

### 自定义函数

```shell
[ function ] funname[()]
{
	Action;
	[return int;]
}
```

shell脚本是逐行运行，不会编译。所以必须在调用函数的地方之前，先声明函数

函数返回值，只能通过 `$?` 系统变量获得。若不加return，将以最后一条命令的运行结果作为返回值。return后跟数值n（0-255）

## *文本处理工具*

在Linux中，grep、sed、awk等文本处理工具都支持通过正则表达式来进行模式匹配