### Architecture of a Python Web



[What is Gunicorn?. The standard web servers such as… | by Serdar İlarslan | Medium](https://medium.com/@serdarilarslan/what-is-gunicorn-5e674fff131b)

[Deploying to Production — Flask Documentation (3.1.x)](https://flask.palletsprojects.com/en/stable/deploying/)



Flask 自带的服务器是单线程的，并且默认不支持处理多个请求的并发。这意味着它只能在同一时间处理一个客户端的请求。相比之下，Gunicorn 是一个预先分叉（pre-forked）的WSGI服务器，它可以启动多个工作进程来处理并发请求，大大提高了应用的响应能力和吞吐量



[Using SQLAlchemy and Flask to build a simple, data-driven web app | by Chris Morrow | Medium](https://cmmorrow.medium.com/using-sqlalchemy-and-flask-to-build-a-simple-data-driven-web-app-17e2d43778bb)





## *WSGI*

Web Server Gateway Interface, WSGI，是 Python 社区制定的一个 Web 服务器与 Python 应用程序或框架之间的通信接口标准 / 规范。WSGI 旨在为 Python 应用提供一种简单且通用的机制，以促进 Python Web 框架应用和多种 Web 服务器之间的移植性

在 WSGI 出现之前，Python 的 Web 应用通常依赖于特定的 Web 服务器软件，这意味着不同的 Web 应用和框架很难在不同的服务器之间进行移植。WSGI 的推出解决了这个问题，它定义了一个清晰的规范，使得任何遵循 WSGI 的 Web 应用都能够在任何支持 WSGI 的 Web 服务器上运行

WSGI 提供了一种标准化的方式让 Web 应用和 Web 服务器进行通信，这意味着：

- 开发人员可以选择他们喜欢的任何 WSGI 兼容的 Web 框架来编写应用，如 Flask、Django 等
- 当需要部署时，可以选择最佳的服务器软件（如 Gunicorn、Nginx、uWSGI、Apache mod_wsgi 等），而无需修改应用代码

### 组件

WSGI 规范主要定义了两方面的内容：

1. **WSGI 服务器**（或“网关”）：负责处理原始的 HTTP 请求、解析环境信息以及组装输入数据并传递给应用程序
2. **WSGI 应用程序**：是一个可调用对象（通常是一个函数或者一个类实例），它接收两个参数，一个表示环境信息的字典 `environ`，以及一个可调用的响应开始函数 `start_response`。应用程序使用这些信息来构建 HTTP 响应

### WSGI 应用示例

Python 内置了一个叫做 wsgiref 的 WSGI 服务器模块，它是用纯 Python 编写的 WSGI 服务器的参考实现。所谓参考实现是指该实现完全符合 WSGI 标准，但是不考虑任何运行效率，仅供开发和测试使用

下面是一个非常基础的 WSGI 应用示例：

```python
# hello.py

def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return [b'<h1>Hello, web!</h1>']
```

```python
# server.py
# 从wsgiref模块导入:
from wsgiref.simple_server import make_server
# 导入我们自己编写的application函数:
from hello import application

# 创建一个服务器，IP地址为空，端口是8000，处理函数是application:
httpd = make_server('', 8000, application)
print('Serving HTTP on port 8000...')
# 开始监听HTTP请求:
httpd.serve_forever()
```

在这个示例中，`application` 函数就是一个 WSGI 应用。该函数接收两个参数：`environ` 字典包含了请求的所有信息，而 `start_response` 是一个回调函数，用来发送响应状态和响应头。最后返回的是一个响应体的字节串列表

### WSGI 中间件

WSGI 还允许开发者创建中间件，即可以同时充当服务器和应用程序的组件。中间件可以处理请求、响应或者两者，然后将其传递给下一个 WSGI 组件

例如，一个中间件可能会处理身份验证、日志记录、请求/响应修改等任务

### WSGI 工具库

WSGI 工具库提供了用于开发符合 WSGI 标准的 web 应用的工具和功能。工具库中可能包括：

- Request 和 Response 对象封装：简化了 HTTP 请求和响应的处理
- 中间件组件：在请求/响应流程中提供钩子用于执行额外的逻辑，如会话管理、认证等
- 服务启动和管理：帮助开发者启动 WSGI 应用，并与多种 web 服务器集成
- 实用工具函数：例如 URL 路由解析、模板渲染、表单数据处理等

一些流行的 WSGI 工具库包括 Werkzeug（作为 Flask 的基础组件）和 WebOb 等

## *Flask 框架*

用 Python 开发一个 Web 框架十分容易，所以 Python 有上百个开源的 Web 框架。这章我们介绍一个很常用的 Python 微框架 Flask

[Flask 入门教程 (helloflask.com)](https://tutorial.helloflask.com/)

```cmd
$ pip3 install flask
```

### 路由装饰器

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/hello/<name>')
def hello(name):
    # 使用渲染模板返回个性化的问候信息
    return render_template('hello.html', name=name)

if __name__ == '__main__':
    app.run(debug=True)
```

上面的代码创建了一个最简单的 Flask 应用。我们使用 `app.route()` 装饰器告诉 Flask，当用户访问应用的根 URL（即 `'/'`）时，应该调用 `index` 函数，并返回 `'Hello, World!'` 字符串。这个字符串会显示在用户的浏览器中

* 通过 `Flask()` 构造器返回 WSGI 句柄对象
* Flask 通过 Python 装饰器 `@app.route()` 内部自动地把某个 URL 和视图函数给关联起来。每个视图函数返回的内容可以是 HTML 页面，也可以是其他类型的响应。这种映射关系定义了 Flask 应用的路由

`@app.route` 可以接收多个参数

- **rule**：这是与装饰器关联的 URL 规则（即路径），它是一个字符串，必须以 `/` 开头。这里的 URL 是相对 URL（或内部 URL），即不包含域名的 URL。例如：`@app.route('/about')` 会将函数映射到网站的 `/about` 路径上
- **methods**：这是一个列表，用来指定视图函数可以响应的 HTTP 方法（如 `GET`, `POST`, `PUT`, `DELETE` 等）。**默认情况下，路由只会响应 `GET` 请求**。如果想处理其他类型的请求，需要相应地设置这个列表。例如：`@app.route('/submit', methods=['POST'])` 表明 `submit` 函数将响应 POST 请求
- **endpoint**：Flask 内部使用端点（endpoint）名字来唯一地标识一个视图函数，如果不显式地指定，Flask 默认使用视图函数的名称作为端点名
- **strict_slashes**：如果设置为 `False`，对于同一个路由，Flask 将不区分末尾斜杠的有无。例如，`/about` 和 `/about/` 会被认为是相同的路径
- **redirect_to**：如果设置了这个参数，访问这个路由会立即重定向到提供的地址
- **defaults**：可以为视图函数的参数提供默认值，它接受一个 dict，里面存储着 URL 中动态变量的映射关系。这在创建 URL 中包含可选参数时非常有用
- **subdomain**：指定该路由仅适用于特定的子域。这允许同一应用为不同的子域提供不同的视图函数
- **host**：类似于 `subdomain`，但它允许你为整个主机名指定路由
- **provide_automatic_options**：是否自动添加 `OPTIONS` HTTP 方法作为视图函数支持的方法之一。默认情况下，Flask 会自动管理

### View Functions

视图函数 view functions 是 Flask 应用中的一个核心概念，它用于响应 requests，即它是**与特定 URL 规则关联的函数**。当访问与该函数关联的 URL 时，Flask 将执行视图函数，并将其返回值作为响应发送给客户端。视图函数负责处理 HTTP 请求并根据请求内容生成相应的 HTTP 响应

在一个典型的 Flask 应用中，视图函数通常会做以下几件事：

1. 获取 HTTP 请求信息：通过 `flask.request` 对象获取请求方法、表单数据、查询参数、cookies 等
2. 执行业务逻辑：处理数据、执行计算或者与数据库交互等操作
3. 返回响应：返回字符串、生成 HTML 页面，或者使用 `jsonify` 返回 JSON 数据，以及设置状态码、重定向、设置 cookies 等

### `flask.requests`

`flask.request` 是 Flask 框架中代表当前请求的对象，它封装了客户端发出的 HTTP 请求的内容。每当一个请求到达 Flask 应用时，Flask 会创建一个 `request` 对象，其中包含了这个 HTTP 请求的所有信息。这个对象只在函数处理请求的过程中有效，它是上下文局部的，这意味着可以在不同的线程中安全地访问相同的请求

[flask.Request — Flask API](https://tedboy.github.io/flask/generated/generated/flask.Request.html)

### render_template

`render_template` 是一个函数，通常用于 Python 的 Flask web 框架中。它用于将你的 HTML 模板文件与后端的 Python 数据结合起来，并生成最终的 HTML 响应内容发送给客户端（比如用户的浏览器）。

在 Flask 应用程序中，`render_template` 函数位于 `flask` 模块中，负责读取模板文件（通常是 Jinja2 模板），填充其中的动态内容，并返回渲染好的 HTML 字符串。这样可以使得在服务器端创建动态网页变得简单。

使用 `render_template` 时，通常会提供模板的名字和需要传递给模板的任何参数作为关键字参数。例如：

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    user_name = "Visitor"
    return render_template('index.html', name=user_name)

if __name__ == '__main__':
    app.run(debug=True)
```

在上述例子中，`'index.html'` 是存放在 Flask 应用的 `templates/` 目录下的模板文件。`render_template` 会渲染此模板，并使用 `user_name` 变量的值替换模板中相应的占位符。最终返回的是经过渲染的 HTML 页面，其内容是基于 `index.html` 模板并插入了 "Visitor" 作为用户名的结果。

## *Dash*

Dash 是一个开源的 Python web 应用框架，专门为创建交互式的 web 分析应用程序而设计。它是由 Plotly 公司开发的，并建立在 Flask、Plotly.js 和 React.js 之上。Dash 旨在使数据科学家和分析师能够轻松地构建具有复杂用户界面（UI）元素的数据可视化界面，而无需深入了解前端技术

# 模板



本章展示的是 Jinja2 模板引擎



Jinja2 是一个非常流行的模板引擎 template engine，用于 Python 编程语言。它提供了一种表达式和控制结构来动态地生成 HTML 或其他标记语言文档。Jinja2 的语法非常类似于 Python

### 分隔符

Jinja2 有三种分隔符 delimiter

* `{# ... #}` 表示注释

  ```jinja2
  {# This is a comment and won't be rendered. #}
  ```

* `{{ ... }}` 双大括号用来输出变量或表达式的结果

* `{% ... %}` 大括号百分号用于执行语句，比如循环、条件判断或定义块

## *变量*

### 变量定义

在 Jinja2 模板中，定义变量通常是通过从后端框架（例如 Flask 或 Django）传递上下文数据到模板来实现的。不过，在模板内部，也可以使用 `{% set %}` 标签定义或重新赋值变量

以下是在 Jinja2 中定义和使用变量的几种方式

* 从后端框架传递变量

  后端应用通常向模板渲染函数传递一个上下文对象，该对象包含可在模板中使用的变量。以 Flask 为例

  ```python
  @app.route('/')
  def index():
      # 定义 Python 变量
      user_name = 'Alice'
      # 将变量作为上下文传递给模板
      return render_template('index.html', user_name=user_name)
  ```

  在 `index.html` 模板中，可以这样使用这个变量

  ```jinja2
  <p>Hello, {{ user_name }}!</p>
  ```

* 在模板中定义变量

  可以直接在模板中使用 `{% set %}` 来定义变量

  ```jinja2
  {% set user_age = 30 %}
  <p>The user is {{ user_age }} years old.</p>
  ```

### 作用域

## *控制流*

### 条件控制

使用 `{% if %}`、`{% elif %}` 和 `{% else %}` 来进行条件判断

```jinja2
{% if user.is_authenticated %}
  <p>Welcome back, {{ user.name }}!</p>
{% else %}
  <p>Please log in to continue.</p>
{% endif %}
```

### 循环

使用 `{% for %}` 和 `{% endfor %}` 进行循环迭代

```jinja2
<ul>
{% for item in item_list %}
  <li>{{ item }}</li>
{% endfor %}
</ul>
```

### 宏

Jinja2 里没有函数，用宏可以用于定义可以重用的模板片段

* 定义宏

  下面的宏定义创建了一个自定义的 `input` HTML 元素，接受参数 `name`、`value` 和 `type`

  ```jinja2
  {% macro input(name, value='', type='text') %}
  <input type="{{ type }}" name="{{ name }}" value="{{ value | escape }}">
  {% endmacro %}
  ```

* 调用宏

  ```jinja2
  {{ input('username') }}
  {{ input('email', type='email') }}
  ```

* 导入宏

  如果有多个模板，并且想在不同模板中重用相同的宏，可以将其放入单独的文件中，并在需要时导入它们

  假设宏定义在名为 `_macros.html` 的文件中

  ```jinja2
  {% from '_macros.html' import input %}
  ```

## *模块*

### block

块（blocks）是可以在子模板中重写的部分，这些块在父模板中被定义，并可以在子模板中填充。例如，在 `base.html` 中

```jinja2
<title>{% block title %}Default Page Title{% endblock %}</title>
```

然后在子模板中重写

```jinja2
{% block title %}Page Title{% endblock %}
```

### extends

继承允许基于一个基础模板来创建子模板，通过 `{% extends 'base.html' %}` 声明。例如

```jinja2
{% extends "base.html" %}
```

## *过滤器*

过滤器可以修改变量的显示方式，通过管道符号 `|` 应用。例如，使用 `capitalize` 过滤器将字符串首字母大写

```jinja2
<p>{{ user_name|capitalize }}</p>
```

多个过滤器可以通过 `|` 链式调用

### 内置过滤器

* safe
* capitalize
* lower
* upper
* title
* trim
* striptags
* escape（或使用缩写 e）
* replace
* default
* length
* jsonify
* sort

### 自定义过滤器

除了使用内置过滤器之外，还可以创建自定义过滤器。这通常在 Python 代码中完成，并添加到 Jinja2 环境中

```python
def reverse(s):
    return s[::-1]

# 添加自定义过滤器到 Jinja2 环境
env.filters['reverse'] = reverse
```

在模板中使用自定义过滤器

```jinja2
{{ "olleh" | reverse }}
```



# 静态文件