## README

### 项目名称

Distributed-Performance-Monitor on Linux

### 项目简介

一个分布式Linux性能分析监控平台，通过 docker 快速构建。使用 gRPC 和 protobuf 作为通信协议

### 目录结构

```
/project
├── CMakeLists.txt
├── docker   # dockerfile & docker scripts to run and get into container
├── LICENSE
├── monitor  # processing infos from /proc/*
├── protobuf # communication protocol (includes gRPC)
├── README.md
└── rpc      # rpc client & server
```

### 构建镜像和编译

使用 Docker 构建项目，确保系统上已经正确安装了 docker，下面的命令假设你的 docker 已经拥有了 sudo 权限

``` cmd
$ cd /path/to/project/docker/build
$ docker build --network host -f base.dockerfile .
```

镜像 build 好之后就可以运行容器了

```cmd
$ docker tag <image_id> linux:monitor
$ cd /path/to/project/docker/scripts
# 启动容器
$ ./monitor_docker_run.sh 
# 进入容器
$ ./monitor_docker_into.sh
```

进入容器后还需要编译一下代码

```cmd
$ cd work
$ cd cmake
$ cmake ..
$ make -j6
```

### 运行

假设使用的三个终端都已经通过 `./monitor_docker_into.sh` 进入了容器

1. 启动 rpc server 收集数据

   ```cmd
   $ cd rpc_manager/server
   $ ./server
   ```

2. 新建终端，启动 monitor (rpc client)

   ```cmd
   $ cd work/cmake/test_monitor/src
   $ ./monitor
   ```

3. 新建终端，启动 Qt GUI

   ```cmd
   $ cd work/cmake/display_monitor
   $ ./display 
   ```

### 依赖项

列出项目的所有依赖项，包括库和工具

```
- CMake (>=3.26)
- abseil (>=20200225.2)
- protobuf (>=3.14.0)
- gRPC (>=1.30.0)
```

### 许可证

明确项目使用的许可证。

### 联系方式

Email: wj.feng@tum.de



## 面试

环境构建

Dockerfile的编写





