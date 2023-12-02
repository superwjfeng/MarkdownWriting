iniparser 是一个 C/C++ 用于解析 INI 文件的库。INI 文件是一种简单的文本文件格式，通常用于配置文件，其中包含了以键值对形式存储的数据。

### 安装

[ndevilla/iniparser: ini file parser (github.com)](https://github.com/ndevilla/iniparser)

```cmd
$ git clone https://github.com/ndevilla/iniparser.git
$ make
```

iniparser 不支持 `make install`，我们手动将make生成的动静态库 `libiniparser.a` 和 `libiniparser.so.l` 放到 lib 下供编译的时候链接用

### 使用

这里是如何在 C 语言项目中使用 `iniparser` 的基本步骤：

1. 包含头文件

   ```c
   #include "iniparser.h"
   ```

2. 加载 INI 文件

   1. 使用 `iniparser_load` 函数来parse INI 文件。这个函数返回一个 `dictionary` 对象，这是 `iniparser` 用于存储解析后的数据的结构
   2. 使用 `iniparser_dump` 函数将一个 dictionary dump到一个fd（可选）

   ```c
   dictionary *ini = iniparser_load("yourfile.ini");
   iniparser_dump(ini, stderr);
   ```

3. 读取数据：一旦加载了 INI 文件，就可以使用诸如 `iniparser_getstring`, `iniparser_getint`, `iniparser_getdouble`, `iniparser_getboolean` 等函数来读取特定的值

   ```c
   int value = iniparser_getint(ini, "section:key", default_value);
   ```

4. 释放资源：在完成了 INI 文件的解析之后，记得使用 `iniparser_freedict` 函数来释放与 `dictionary` 对象相关联的资源

   ```c
   iniparser_freedict(ini);
   ```

5. 错误处理：可以考虑在代码中添加适当的错误处理，比如检查文件是否成功加载，以及所请求的键值对是否存在

### 例子

```c++
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "iniparser.h"

void create_example_ini_file(void);
int  parse_ini_file(char * ini_name);

int main(int argc, char * argv[])
{
    int     status ;

    if (argc<2) {
        create_example_ini_file();
        status = parse_ini_file("example.ini");
    } else {
        status = parse_ini_file(argv[1]);
    }
    return status ;
}

void create_example_ini_file(void)
{
    FILE    *   ini ;

    if ((ini=fopen("example.ini", "w"))==NULL) {
        fprintf(stderr, "iniparser: cannot create example.ini\n");
        return ;
    }

    fprintf(ini,
    "#\n"
    "# This is an example of ini file\n"
    "#\n"
    "\n"
    "[Pizza]\n"
    "\n"
    "Ham       = yes ;\n"
    "Mushrooms = TRUE ;\n"
    "Capres    = 0 ;\n"
    "Cheese    = Non ;\n"
    "\n"
    "\n"
    "[Wine]\n"
    "\n"
    "Grape     = Cabernet Sauvignon ;\n"
    "Year      = 1989 ;\n"
    "Country   = Spain ;\n"
    "Alcohol   = 12.5  ;\n"
    "\n");
    fclose(ini);
}


int parse_ini_file(char * ini_name)
{
    dictionary  *   ini ;

    /* Some temporary variables to hold query results */
    int             b ;
    int             i ;
    double          d ;
    const char  *   s ;

    ini = iniparser_load(ini_name);
    if (ini==NULL) {
        fprintf(stderr, "cannot parse file: %s\n", ini_name);
        return -1 ;
    }
    iniparser_dump(ini, stderr);

    /* Get pizza attributes */
    printf("Pizza:\n");

    b = iniparser_getboolean(ini, "pizza:ham", -1);
    printf("Ham:       [%d]\n", b);
    b = iniparser_getboolean(ini, "pizza:mushrooms", -1);
    printf("Mushrooms: [%d]\n", b);
    b = iniparser_getboolean(ini, "pizza:capres", -1);
    printf("Capres:    [%d]\n", b);
    b = iniparser_getboolean(ini, "pizza:cheese", -1);
    printf("Cheese:    [%d]\n", b);

    /* Get wine attributes */
    printf("Wine:\n");
    s = iniparser_getstring(ini, "wine:grape", NULL);
    printf("Grape:     [%s]\n", s ? s : "UNDEF");

    i = iniparser_getint(ini, "wine:year", -1);
    printf("Year:      [%d]\n", i);

    s = iniparser_getstring(ini, "wine:country", NULL);
    printf("Country:   [%s]\n", s ? s : "UNDEF");

    d = iniparser_getdouble(ini, "wine:alcohol", -1.0);
    printf("Alcohol:   [%g]\n", d);

    iniparser_freedict(ini);
    return 0 ;
}
```