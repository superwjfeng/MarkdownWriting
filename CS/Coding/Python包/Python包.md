本文档是一些比较复杂的Python包的使用笔记

# argparse

[argparse --- 用于命令行选项、参数和子命令的解析器 — Python 3.12.3 文档](https://docs.python.org/zh-cn/3/library/argparse.html)

argparse 是 Python 标准库中的一个模块，用于编写用户友好的命令行界面。程序定义它需要的参数，argparse 将会从 `sys.argv` 中解析出那些参数。argparse 模块还会自动生成帮助和使用手册，并在用户给程序传递无效参数时报错

### 基本用法

1. 创建 `ArgumentParser` 对象

   ```python
   import argparse
   parser = argparse.ArgumentParser(description='这是一个示例程序.')
   ```

2. 使用 `add_argument` 方法添加参数

   [python之parser.add_argument()用法——命令行选项、参数和子命令解析器-CSDN博客](https://blog.csdn.net/qq_34243930/article/details/106517985)

   * name or flags：选项字符串的名字或者列表，例如foo或者 `-f,--foo`
   * action：命令行遇到参数时的动作，默认值是store
   * store_const：表示赋值为const
   * append,将遇到的值存储成列表,也就是如果参数重复则会保存多个值;
   * append_const：将参数规范中定义的一个值保存到一个列表;
   * count：存储遇到的次数。此外也可以继承 `argparse.Action` 自定义参数解析
   * nargs-应该读取的命令行参数个数,可以是具体的数字,或者是?号,当不指定值时对于Positional argument使用default,对于Optional argument使用const;或者是*号,表示0或多个参数;或者是+号表示1或多个参数
   * const：action和nargs所需要的常量值
   * default：不指定参数时的默认值
   * type：命令行参数应该被转换成的类型
   * choices：参数可允许的值的一个容器
   * required-可选参数是否可以省略(仅针对可选参数)。
   * help：参数的帮助信息。当指定为 `argparse.SUPPRESS` 时表示不显示该参数的帮助信息
   * metavar：在usage说明中的参数名称。对于必选参数默认就是参参数名称，对于可选参数默认是全大写的参数名称
   * dest：解析后的参数名称，若未给出，则从选项中推断出来。默认情况下，对于可选参数选取最长长的名称，中划线转换为下划线

   ```python
   parser.add_argument('echo', help='描述这个参数的作用')
   parser.add_argument('-v', '--verbose', action='store_true', help='增加输出的详细性')
   ```

   - 第一个参数通常是位置参数（如 `'echo'`）

   - `-v` 或 `--verbose` 表示可选参数

3. 使用 `parse_args()` 解析添加参数的参数对象，获得解析对象

下面是一段完整的代码示例：

```python
import argparse

# 初始化解析器
parser = argparse.ArgumentParser(description="这是一个示例程序.")
# 添加位置参数
parser.add_argument("echo", help="echo 参数的帮助信息")
# 添加可选参数
parser.add_argument("-v", "--verbose", action="store_true", help="显示更多信息")

# 解析参数
args = parser.parse_args()

# 根据参数执行不同的操作
if args.verbose:
    print(f"你输入了 '{args.echo}', 并且激活了详细模式.")
else:
    print(args.echo)
```

### 选项类型

[[编程基础\] Python命令行解析库argparse学习笔记_python argparse output-CSDN博客](https://blog.csdn.net/LuohenYJ/article/details/109397999)

* 可选参数
* 必需参数 设定 `required = True`
* 位置参数 positional 不需要前缀符号，直接输入参数值即可

### 默认值问题

如果在使用 `argparse` 的 `add_argument` 方法时没有指定 `default` 参数，那么默认值的行为依赖于该参数是否为可选（optional）或位置（positional）参数：

* **对于可选参数**（通常以 `-` 或 `--` 开头），如果命令行中没有给出该参数，则默认值为 `None`
* **对于位置参数**（那些不以 `-` 或 `--` 开头的参数），必须在命令行中提供这个参数，否则 `argparse` 会报错。不存在默认值的概念，因为它们是必须由用户明确提供的

### 参数列表的覆盖

```python
arguments = ['-m', '0', 
           '-c', case_path,
           '-e', extend_path,
           '-r', root_path, 
           '-u', master_uri]
if not flag:
    arguments += ['-m', '1']	
```

arguments 是一个列表，当执行 `+=` 操作时，会将右侧的列表元素追加到左侧的列表中。这种操作不会替换或覆盖原有元素，而是在列表末尾添加新元素

这意味着如果代码后面执行了 `arguments += ['-m', '1']`，它会在列表的末尾添加 `'-m'` 和 `'1'` 这两个元素

因此，原始列表中包含的 `'-m', '0'` 不会被删除或更改，但现在列表中会有两组 `'-m'` 和对应值。这可能导致问题，因为通常情况下命令行解析器（如 `argparse`）会根据参数的最后一次出现来解析值。在这种情况下，如果传递整个 `simulator_arguments` 列表给解析器，`'-m'` 的值可能会被解析为 `'1'` 而不是 `'0'`，因为它是最后出现的



## *action*

`argparse.Action` 用于定义解析某个命令行参数时应该执行的动作。每个 `Action` 都代表了一种当特定参数被命令行接收时要进行的操作

注意：action是用来检查参数通常用于在解析某个特定参数时立即触发某种行为，比如转换值、检查依赖关系或者触发辅助动作，如果是用来检查多个参数之间的关系，应该是在参数解析之后进行检查

### 内置action

当在使用 `argparse.ArgumentParser` 的 `add_argument` 方法添加命令行参数的时候，可以通过 `action` 参数来指定不同的动作。`argparse` 模块内置了多种标准动作，例如：

- `store`：这是默认动作，会将命令行参数的值保存起来。如果不指定 `action` 的话，默认会使用这个动作

- `store_const`：保存一个被定义为参数的常量值，而不是用户输入的值

- `store_true` / `store_false`：这两种动作用于布尔开关。如果命令行中存在对应的参数，则分别直接赋值 `True` 或 `False`，而不需要显式给出 `=False` 或 `=True`

  如果给出了这个action，则不应该再给出 type 和 choices，因为这个action已经限定了参数的类型和可选值

- `append`：将值添加到列表中。如果参数重复出现，则每次都添加新的元素

- `append_const`：将一个定义好的常量添加到列表中，类似于 `append`，但保存的是常量而非用户输入的值

  - `count`：计算某个参数出现的次数，通常用于增加日志级别的场景

- `help`：显示参数的帮助信息并退出

- `version`：显示程序的版本信息并退出

### 自定义action

1. 继承 `argparse.Action`
2. （必须）重写覆盖 `__call__`，该方法接收四个参数：`parser`（解析器对象）、`namespace`（要填充的属性对象）、`values`（命令行参数），以及 `option_string`（选项字符串）
   1. 在 `__call__` 方法中处理参数并设置 `namespace` 对象的属性，以便之后可以通过这些属性访问对应的参数值
3. （可选）重写覆盖 `__init__` 和 `format_usage` 等方法

```python
class FooAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        print('%r %r %r' % (namespace, values, option_string))
        setattr(namespace, self.dest, values)
```









### 进阶功能

除了上述功能，`argparse` 还支持很多其他功能，比如：

- 为参数指定数据类型
- 选择性地使参数成为必须提供项 (`required=True`)
- 设置默认值 (`default=` 值)
- 限制参数的选择范围 (`choices=[...]`)
- 支持变长参数列表 ( `nargs='+'` 或 `'*'` 等)

# SymPy

[SymPy 1.13.1 documentation](https://docs.sympy.org/latest/index.html)

CASs, Computer Algebra Systems

> A CAS is a package comprising a set of algorithms for performing symbolic manipulations on algebraic objects, a language to implement them, and an environment in which to use the language.

## *Symbols & Expressions*

### 定义Symbols

```python
sympy.core.symbol.symbols(names, *, cls=<class 'sympy.core.symbol.Symbol'>, **args) → Any
```

SymPy中使用变量之前必须要定义，使用 `symbols()` 来定义变量。`symbol()` 输入可以用 `,` 或者空格分隔的一个string

```python
x, y, z = symbols('x y z')
```

如果想定义一系列连续的符号，可以在字符串中使用冒号 `:` 标记来指示这个范围

```python
x1, x2, x3, x4, x5 = symbols('x1:x6')
```

因此要注意的是，如果原string中存在冒号会造成解析错误

### 将Strings转换为Expressions

```python
sympy.core.sympify.sympify(a, locals=None, convert_xor=True, strict=False, rational=False, evaluate=None)
```



```python
>>> str_expr = "x**2 + 3*x - 1/2"
>>> expr = sympify(str_expr)
```

### Expression操作

* `subs`
* `evalf`
* `lambdify`

## *SymPy中特殊的操作符*

### `==`

structural equality

### `^`

### `/`

## *Parsing*

### Parser

* `parse_expr()` 用于将一个string转换为一个SymPy表达式
* `stringify_expr()` 用于将一个string转换为Python代码

### Transformation

Transformation 指的是一个接受参数 `tokens, local_dict, gloabl_dict` 并返回一个transformed tokens的列表，一般用来传递给 `parse_expr()`

# PyParsing

[Welcome to PyParsing’s documentation! — PyParsing 3.1.1 documentation (pyparsing-docs.readthedocs.io)](https://pyparsing-docs.readthedocs.io/en/latest/)

注意：在pyparsing 3.0中，许多最初使用驼峰式命名（camelCase）的方法和函数名已经被转换为符合PEP8规范的下划线分隔式命名（snake_case）。比如 `parseString()`正被重命名为`parse_string()`，`delimitedList`被更改为`DelimitedList_`等等。可能会在legacy edition中看到旧的名称，它们将通过同义词被兼容一段时间，但这些同义词将在未来的版本中移除

<img src="pyparsingClassDiagram_3.0.9.jpg">

## *Steps to Follow*

1. 定义词法：首先定义要匹配的tokens和pattern，并将其赋值给一个变量。此时也可以定义可选的结果名称或解析动作
2. 在这个变量上调用`parse_string()`、`scan_string()`或`search_string()`方法，并传入要解析的字符串。在匹配过程中，默认情况下会跳过令牌之间的空白（尽管可以更改这个设置）。当发生令牌匹配时，任何定义的解析动作方法都会被调用
3. 处理返回的ParseResults对象中的解析结果。ParseResults对象可以像访问字符串列表一样被访问。如果在定义令牌模式时使用 `set_results_name()` 定义了名称，还可以通过返回结果的命名属性来访问匹配结果

## *ParserElement*

### 类型

* Literal & CaselessLiteral
* Keyword & CaselessKeyword
* Word
* Char & CharsNotIn
* Regex

### 方法

* `scan_string()`
  * 返回一个list of triples `[(matched_tokens, start location, end location)]`

### Expression Subclasses

### Expression Operators

### Converter Subclasses

* Combine
* Supress 