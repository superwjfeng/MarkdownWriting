

## *项目架构*

### 文件组织架构

conf存放配置文件

git存放从git上clone下来的第三方库

src中存放项目源代码（包括头文件和cpp源文件）

test存放单元测试文件

third存放第三方库

lib存放本项目输出的库

doc存放一些README中用到的图片和中文版README



### 配置文件



## *项目组件*

### initparser

[ndevilla/iniparser: ini file parser (github.com)](https://github.com/ndevilla/iniparser)

```cmd
$ cd path/to/git
$ git clone https://github.com/ndevilla/iniparser.git
$ make
```

iniparser 不支持 `make install`，我们手动将make生成的动静态库 `libiniparser.a` 和 `libiniparser.so.l` 放到 `third/lib/iniparser/` 下

同时将iniparser中的两个头文件也放到 `third/include/iniparser` 下

为了使用initparser，我们为其封装了initconfig.h和configdef.h两个头文件

### log4cpp

<https://log4cpp.sourceforge.net>

```cmd
$ wget https://nchc.dl.sourceforge.net/project/log4cpp/log4cpp-1.1.x%20%28new%29/log4cpp-1.1/log4cpp-1.1.3.tar.gz
$ tar xzvf log4cpp-1.1.3.tar.gz
$ cd log4cpp
$ ./configure --prefix=/home/${USER}/SharedBike/third
$ make
$ make check
$ make install
```

指定了prefix会安装到third下面，log4cpp的头文件和库会分别被安装在third下面的include和lib文件下中

### gtest

### memcheck

