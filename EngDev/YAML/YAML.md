[YAML Ain’t Markup Language (YAML™) Version 1.1](https://yaml.org/spec/1.1/#id857168)

## *数据类型*

### Collections 集合

在YAML中，集合指的是可以包含多个元素的复合数据类型。YAML支持两种类型的集合：**序列 sequences** 和 **映射mappings**

* **Sequences**用于表示数组或列表，通过使用短横线`-`标记来创建。每个短横线后面跟着**空格**，然后是该项目的值

  ```yaml
  # 序列示例
  - Apple
  - Banana
  - Cherry
  ```

* **Mappings**用来表示键值对，类似于字典或哈希表。在映射中，每个键后面都跟着一个冒号和一个空格，然后是相应的值

  ```yaml
  # 映射示例
  name: John Doe
  age: 30
  married: true
  ```

### Structures  结构

YAML文件通常由上述的集合构成，并可以通过嵌套这些集合来创建更复杂的结构。例如，可以在一个映射中嵌套一个序列，反之亦然

```yaml
# 结构示例
employees:
  - name: John Doe
    role: Manager
  - name: Jane Smith
    role: Developer

address:
  street: "123 Main St"
  city: "Anytown"
  zip: 12345
```

### Scalars 标量

标量是最基本的数据单元，代表单个的值，比如字符串、布尔值、整数或浮点数。在YAML中，标量可以直接表示，也可以用引号（单引号或双引号）包围起来。使用引号的好处是可以包含特殊字符或保留字。例如：

```yaml
# 标量示例
integer: 123
boolean: true
string: "Hello, World!"
special_characters: "Newline character is \n"
```

* 布尔值、数字、日期等直接写，不要加引号

* 字符串

  * **单引号（Single quoted）**：在单引号内的字符串会按照字面意思来解释。特殊字符不需要转义，除了单引号本身，你需要用两个单引号表示一个单引号。

    ```yaml
    single_quoted: 'Here''s a string with a single quote (apostrophe).'
    ```

  * **双引号（Double quoted）**：双引号允许你使用转义序列，比如换行符 `\n` 或者 Unicode 字符。这使得双引号适合包含特殊字符的字符串。

    ```yaml
    double_quoted: "Line1\nLine2"
    ```

* 布尔值：在YAML（YAML Ain't Markup Language）中，某些字符串被认为是特殊的，并且会被解析为布尔值。`off`, `on`, `yes`, `no`, `true`, `false`, `y`, `n` 等都能够被识别并转换为布尔值。所以在 YAML 文件中使用 `off` 时，它通常会被解释为布尔值 `False`

### Tags 标签

YAML提供了扩展的数据类型，称为tags。Tags允许你显式地指定一个值的数据类型，或者使用自定义数据类型。在YAML中，标签以`!`开头，后面跟着标签名。例如，你可以使用`!!str`来强制将一个值解析为字符串，即使它看起来像一个数字。自定义标签可以与应用程序相关，以便自定义处理某些数据。例如：

```yaml
# 标签示例
explicit_string: !!str 1234
date: !!timestamp "2023-01-01T12:00:00Z"
custom_tag: !mytype { name: John Doe, age: 30 }
```

在这个例子中，`1234`通常会被视为一个整数，但标签`!!str`告诉解析器将其作为字符串处理。同样地，`!mytype`标签可以用来指示解析器使用自定义逻辑来处理该值

## *过程*

<img src="YAML_Overview.png">