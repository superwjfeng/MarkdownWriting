ANTLR, ANother Tool for Language Recognition 是一个强大的解析器生成器，它可以用来读取、处理、执行或翻译结构化文本或二进制文件。它广泛用于构建语言、工具和框架。ANTLR 由 Java 编写，能够生成适用于包括 Java、C#、Python、JavaScript 和 Go 在内的多种编程语言的解析器\

使用 ANTLR 工具，我们只需要写出词法和语法分析的 **规范（specification）**， 然后它会帮我们生成 lexer 和 parser 乃至 visitor，非常方便

## *安装 ANTLR*

[Ubuntu 20.04桌面版 安装Antlr4_ubuntu 安装antlr4-CSDN博客](https://blog.csdn.net/drutgdh/article/details/122603220)

首先要安装 Java

对于 Java 用户，最简单的方法通常是下载 ANTLR 的 jar 文件并将其添加到类路径中。也可以使用像 Maven 或 Gradle 这样的构建工具来管理依赖

```cmd
# 使用 Homebrew 安装 (macOS)
brew install antlr

# 使用 Chocolatey 安装 (Windows)
choco install antlr4
```

## *定义语法*

### 语法文件

定义语言的语法是生成语言的最重要步骤。这个语法文件（通常以 `.g4` 扩展名保存， g 代表 grammar，4 代表 ANTLR 的版本号 4）描述了语言的结构，包括词法规则（Token）和语法规则（Parser rules）

在 `.g4` 文件的开头，需要给文件中定义的语法起个名字，这个名字必须和文件名相同

规范文件中，`//` 表示注释，规范是大小写敏感的，字符串常量用单引号括起



ANTLR 约定词法解析规则以大写字母开头。和 bison 类似。ANTLR 使用 `:` 代表 BNF 文法中的 `->` 或 `::=`；同一终结符/非终结符的不同规则使用 `|` 分隔；使用 `;` 表示一条终结符/非终结符的规则的结束



例如，下面是一个简单的语法示例，该语法识别简单的计算表达式：

```antlr
grammar Expr;

expr:   expr ('*'|'/') expr
    |   expr ('+'|'-') expr
    |   INT
    ;

INT :   [0-9]+ ;
WS  :   [ \t\r\n]+ -> skip ;
```

在这个例子中，“expr”是一个解析规则，而“INT”和“WS”是词法规则。

### 生成解析器代码

使用 ANTLR 工具生成特定目标语言的解析器代码。通过命令行运行 ANTLR 并指定语法文件，就可以生成相应的代码。

```cmd
antlr4 test.g4 -Dlanguage=Python3
```

上述命令会针对 Python 3 生成解析器代码，生成下面这些文件

* testLexer.py：这是根据 `test.g4` 中定义的词法规则自动生成的 Python 词法分析器代码。它用于将输入文本分割成一系列的标记（tokens），这些标记由词法规则定义
* testParser.py：这个文件包含了基于 `test.g4` 中的语法规则生成的 Python 语法分析器代码。它用来根据标记流（由词法分析器提供）构建解析树或抽象语法树（AST）
* testListener.py：ANTLR 自动生成了一个默认的 listener 接口，该接口为遍历解析树（parse tree）中的节点提供了一个回调机制。用户可以扩展这个类以实现自己的逻辑
* test.tokens：这个文件包含了从 `test.g4` 文件中生成的所有 token 的列表。通常每一行代表一个标记，格式为 `TOKEN_NAME=token_number`
* testLexer.tokens：类似于 `test.tokens`，但是这个文件专门包含词法分析期间生成的标记
* test.interp 和 testLexer.interp：这两个文件包含了关于解析器和词法分析器内部工作信息的元数据，主要供 ANTLR 的工具使用，以便于 IDE 调试和可视化解析过程

## *遍历AST*

ANTLR 提供了 Listener 和 Visitor 两种模式来完成语法树的遍历，默认生成的是 Listener 模式的代码，如果要生成 Vistor 模式的代码，需要运行选项中加上 `-visitor`，如果要关闭生成 Listener 模式的代码，需要运行选项中加上 `-no-listener`



在使用 ANTLR 生成的代码时，需要定义一个类继承 `BaseListener` 或 `BaseVisitor`，在其中重写遍历到每个节点时所调用的方法，完成从语法树翻译到 IR 的翻译工作



Listener 模式中为每个语法树节点定义了一个 enterXXX 方法和一个 exitXXX 方法，如 `void enterExpr(calcParser.ExprContext ctx)` 和 `void exitExpr(calcParser.ExprContext ctx)`。遍历语法树时，**程序会自动遍历所有节点**，遍历到一个节点时调用 enter 方法，离开一个节点时调用 exit 方法，我们需要在 enter 和 exit 方法中实现翻译工作

Vistor 模式中为每个语法树节点定义了返回值类型为泛型的 visitXXX 方法，如 `T visitExpr(calcParser.ExprContext ctx)`。遍历语法树时，我们需要调用一个 `Visitor` 对象的 `visit` 方法遍历语法树的根节点，`visit` 方法会根据传入的节点类型调用对应的 visitXXX 方法，我们需要在 visitXXX 方法中实现翻译工作。在翻译工作中，我们可以继续调用 `visit` 方法来手动遍历语法树中的其他节点

我们可以发现：Listener 模式中方法没有返回值，而 Vistor 模式中方法的返回值是一个泛型，类型是统一的，并且两种模式中的方法都不支持传参。在我们需要手动操纵返回值和参数时，可以定义一些属性用于传递变量

Listener 模式中会按顺序恰好遍历每个节点一次，进入或者退出一个节点的时候调用你实现的对应方法。而 Vistor 模式中对树的遍历是可控的，我们可以遍历时跳过某些节点或重复遍历一些节点，因此在翻译时推荐使用 Visitor 模式

## *基于 ANTLR 生成的解析器编写代码*

### 编写程序以使用生成的解析器

生成的代码包含了处理输入并创建解析树的所有必要类。可以在应用程序中使用这些类来解析符合定义的语法的输入

例如，如果生成的是 Python 解析器代码，你可能会这样使用它：

```python
from antlr4 import *
from ExprLexer import ExprLexer
from ExprParser import ExprParser

input_stream = InputStream("(1 + 2) * 3")
lexer = ExprLexer(input_stream)
stream = CommonTokenStream(lexer)
parser = ExprParser(stream)
tree = parser.expr()

print(tree.toStringTree(recog=parser))
```

### 处理解析树

你可以访问解析树并编写自己的访问者（Visitor）或监听者（Listener）来遍历树，并根据解析的结构执行特定操作，比如求值、翻译或转换代码。

ANTLR 提供了默认的 Visitor 和 Listener 类，你可以扩展这些类来实现自己的逻辑。

### 总结

ANTLR 是一个非常灵活的工具，用于构建复杂的文本分析任务。以上步骤提供了一个高级概览，但 ANTLR 的能力远不止于此。为了更深入地了解和使用 ANTLR，你可能需要查阅官方文档、教程和书籍，其中详细介绍了如何设计语法、处理解析树以及进行错误处理等高级功能。