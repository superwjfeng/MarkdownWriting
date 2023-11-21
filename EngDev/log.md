# log4cpp

## *介绍*

log4cpp（Log for C++）是一个用于C++编程语言的开源日志记录库，它提供了灵活和高度可配置的日志记录功能。log4cpp的设计灵感来自于Java中的log4j库，它旨在帮助C++开发人员实现高效的日志记录，以便在应用程序中进行调试、跟踪和错误诊断

### 安装

<https://log4cpp.sourceforge.net>

```cmd
$ wget https://nchc.dl.sourceforge.net/project/log4cpp/log4cpp-1.1.x%20%28new%29/log4cpp-1.1/log4cpp-1.1.3.tar.gz
$ tar xzvf log4cpp-1.1.3.tar.gz
$ cd log4cpp
$ ./configure --prefix=${绝对路径}
$ make
$ make check
$ make install
```

## *使用*

以下是使用步骤

1. 包含头文件
2. 初始化日志输出的目的地（appenders）
3. 设置日志输出的格式
4. 设置输出类别（category）和日志优先级（priority）
5. 定义宏 & 使用宏记录日志

### 包含头文件

```c++
#include <log4cpp/Category.hh>
#include <log4cpp/FileAppender.hh>
#include <log4cpp/PatternLayout.hh>
#include <log4cpp/OstreamAppender.hh>
```

### 初始化日志输出的目的地（appenders）

```c
// 输出到 std::cout
log4cpp::Appender *appender = new log4cpp::OstreamAppender("root", &std::cout);
// 输出到 log 文件
//log4cpp::Appender *appender = new log4cpp::FileAppender("root", "test.log");
```

appender 有以下这些：

* **`log4cpp::FileAppender`：输出到文件**
* **`log4cpp::RollingFileAppender`：输出到回卷文件，即当文件到达某个大小后回卷/回滚**
* `log4cpp::OstreamAppender`：输出到一个 ostream 类
* `log4cpp::RemoteSyslogAppender`：输出到远程 syslog 服务器
* `log4cpp::StringQueueAppender`：内存队列
* `log4cpp::SyslogAppender`：本地 syslog
* `log4cpp::Win32DebugAppender`：发送到缺省系统调试器
* `log4cpp::NTEventLogAppender`：发送到 win 事件日志

我们说过日志输出到终端或者文件中实际上是很慢的，因为会引起IO中断，所以我们可以输出到内存里 StringQueueAppender，然后从 StringQueueAppender 输出到其它地方，这样的话线程执行是比较高效的

### 设置日志输出的格式

```c++
log4cpp::PatternLayout *patternLayout = new log4cpp::PatternLayout();
patternLayout->setConversionPattern("%d [%p] - %m%n");
appender->setLayout(patternLayout);``
```

日志输出格式控制有：

* `%%` - a single percent sign 转义字符

* `%c` - the category

* `%d` - the `date\n` Date format:

  The date format character may be followed by a date formatspecifier enclosed between braces. For example, `%d{%\H:%M:%S,%l}` or `%d{%\d %m %Y %H:%\M:%S,%l}` 

  If no date format specifier is given then the following format is used: "Wed Jan 02 02:03:55 1980". The date format specifier admits the same syntax as the ANSI C function strftime, with 1 addition. The addition is the specifier `%l` for milliseconds, padded with zeros to make 3 digits

* `%m` - the message

* `%n` - the platform specific line separator

* `%p` - the priority

* `%r` - milliseconds since this layout was created

* `%R` - seconds since Jan 1, 1970

* `%u` - clock ticks since process start

* `%x` - the NDC

* `%t` - thread name

默认的 ConversionPattern for PatternLayout 被设置为 `%m%n`

### 设置输出类别（category）和日志优先级（priority）

```c++
log4cpp::Category &root = log4cpp::Category::getRoot();
root.setPriority(log4cpp::Priority::NOTICE);
root.addAppender(appender);
```

日志级别总共有：`NOTSET < DEBUG < INFO < NOTICE < WARN < ERROR < CRIT < ALERT < FATAL = EMERG`。日志级别的意思是低于该级别的日志不会被记录

### 定义宏 & 使用宏记录日志

为了避免每次都要写一大堆，可以用一个宏来简化

```c++
#define LOG(__level) log4cpp::Category::getRoot() << log4cpp::Priority::__level << __FILE__ << " " << __LINE__ << ": "
```

* `log4cpp::Category::getRoot()`
* `__FILE__` 和 `__LINE__`

实际使用宏来记录日志

```c++
LOG(DEBUG) << "i am happy.";
LOG(INFO) << "oh, you happy, we happy.";
LOG(NOTICE) << "please do not contact me. ";
LOG(WARN) << "i am very busy now.";
LOG(ERROR) << "oh, what happed?";
```

也可以使用Category定义的函数来简化

```c++
/**
* Log a message with the specified priority.
* @param priority The priority of this log message.
* @param stringFormat Format specifier for the string to write in the log file.
* @param ... The arguments for stringFormat
**/
virtual void log(Priority::Value priority, const char* stringFormat, ...) throw();
/**
* Log a message with the specified priority.
* @param priority The priority of this log message.
* @param message string to write in the log file
**/
virtual void log(Priority::Value priority, const std::string& message) throw();
void debug(const char* stringFormat, ...) throw();
void debug(const std::string& message) throw();
void info(const char* stringFormat, ...) throw();
```

## *工程实践*

### 单例封装

### 日志配置文件

## *spdlog*

# Java

# Python

