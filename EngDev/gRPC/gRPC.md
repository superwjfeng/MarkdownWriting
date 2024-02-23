---
Title: gRPC
Author: Weijian Feng 封伟健
Source: Official documentation for gRPC <https://grpc.io/docs/>, 官方文档中文翻译 <https://doc.oschina.net/grpc?t=56831>
---

# gRPC原理

## *gRPC 实现架构*

https://grpc.io/blog/grpc-stacks/

<img src="gRPC协议栈.png">

### Intro

gRPC（Google Remote Procedure Call）是一种高性能、通用的开源远程过程调用（RPC）框架，由Google开发并开源。它基于HTTP/2协议和Protocol Buffers（protobuf）序列化协议，用于构建跨平台和跨语言的分布式应用程序

gRPC 提供了一种简单的方法来定义服务接口和消息类型（**基于服务，与具体哪种语言无关**），并自动生成支持多种编程语言的客户端和服务器端代码。它支持多种编程语言，如C++、Java、Python、Go、C#等，因此可以在不同的环境和语言之间轻松进行通信

gRPC 在微服务架构和分布式系统中具有广泛的应用。它提供了高效的网络通信和跨语言支持，使得不同服务能够方便地进行通信和协作。无论是构建大规模的云原生应用程序、实现高性能的数据传输，还是构建实时流处理系统，gRPC 都是一个强大的工具

### 基于 HTTP/2

gRPC是基于HTTP/2的再包装，因此其底层调用的仍然是 TCP/IP 协议栈。HTTP/2 提供了双向流、流控制、头部压缩等特性。HTTP/2 的多路复用特性允许同时处理多个请求和响应，提高了网络利用率和性能

既然是 HTTP/2，通信的 Hierarchy 就是帧 frame `->` 信息 message `->` 流 stream `->` 一个 TCP 连接（在 gRPC 中被称为 channel），这些概念对于我们是理解 gRPC 的前提

<img src="基于HTTP2.png" width="80%">

### 基于 protobuf

gRPC 使用 protobuf 作为默认的序列化机制。通过定义服务接口和消息类型的 .proto 文件，gRPC 可以自动生成用于客户端和服务器端的代码。这简化了开发过程，并确保在不同的语言之间保持一致性

## *消息格式*

### 请求消息

```
Request Message: Request Headers -> Length-Prefixed-Message -> End of stream flag
```

请求消息是发起远程调用的消息。gRPC 的请求消息只能由 client 端应用程序触发（虽然 HTTP/2 支持 server 端主动推送消息），它由三个主要部分组成：请求头 Request Headers、长度前缀消息 Length-Prefixed-Message 和流结束标志 End of stream flag (EOS)。client 端首先发送请求头，之后是长度前缀消息，最后是 EOS，标识消息发送完毕

例子可以看这个网站 https://learnku.com/articles/72847

### 响应消息

```
Response Message: Response Headers -> Length-Prefixed-Message (optional) # 数据帧
Response Message: Response Headers -> Length-Prefixed-Message (optional) -> Trailers # 结尾帧
```

响应消息也由三个主要部分组成：响应标头、带长度前缀的消息和尾部。当响应中没有以长度为前缀的消息需要发送给 client 端时，响应消息仅包含响应标头和尾部

与请求消息不同的是，END_STREAM 标志不随数据帧一起发送，它作为一个被称作 Trailers 单独的响应头发送，通知 client 端我们完成了响应消息的发送。Trailers 还会携带请求的状态码和状态消息

# gRPC 的 Client & Server 建立

gRPC的源码库 <https://github.com/grpc/grpc>

## *安装*

https://github.com/grpc/grpc/blob/v1.61.0/src/cpp/README.md

### 使用 Bazel

### 使用 CMake

## *项目结构*

下面的内容以 java & unary rpc 为例

### 大纲

<img src="gRPC流程.drawio.png">

1. api 模块

    定义 protobuf IDL语言并通过命令创建具体的代码，后续client server引入使用

    * message
    * service

    ```protobuf
    syntax = "proto3";
    
    // 后续protobuf生成的java代码一个源文件还是多个源文件xx.java。否则会为每个message、enum、service生成独立的class
    option java_multiple_files = false;
    // 指定protobuf生成的类放置在哪个包中
    option java_package = "com. suns";
    // 指定的protobuf生成的外部类的名字（管理内部类【内部类オ是真正开发使用】）
    option java_outer_classname = "UserServoe";
    
    package com.example;
    
    service HelloService {
        rpc sayHello (HelloRequest) returns (HelloResponse);
    }
    
    message HelloRequest {
        string name = 1;
        // Request中可以有很多个字段（多个函数参数）
    }
    
    message HelloResponse {
        string message = 1;
        // Response中可以有很多个字段（多个函数返回值）
    }
    ```

1. xxxx-server模块

    * 实现api模块中定义的服务接口：新建一个 `xxxServiceImpl.java` 并在里面继承生成的 `xxxImplBase` 类，然后重写实现对应的service
    
      ```java
      public class HelloServiceImpl extends HelloServiceGrpc.HelloServiceImplBase {
          // ...重写实现service方法
      }
      ```
    
    * 发布gRPC服务（创建服务端程序）
    
      ```java
      public class GrpcServer {
          public static void main(String[] args) throws IOException, InterruptedException {
              // 1. 绑定端口
              ServerBuilder serverBuilder = ServerBuilder.forPort(9000);
              // 2. 发布服务
              serverBuilder.addService(new HelloServiceImpl());
              // 3. 创建服务对象
              Server server = serverBuilder.build();
              // 4. 启动服务
              server.start();
              server.awaitTermination();
          }
      }
      ```
    
2. xxxx-client模块
   * 创建服务端stub（代理）
   * 基于stub的RPC调用
   
   ```java
   public class GrpcClient1 {
       public static void main(String[] args) {
           // 1. 创建通信管道
           ManagedChannel managedChannel = ManagedChannelBuilder.forAddress("localhost", 9000).usePlaintext().build();
           // 2. 创建stub 代理对象
           try {
               HelloServiceGrpc.HelloServiceBlockingStub helloService = HelloServiceGrpc.newBlockingStub(managedChannel);
               // 3. 完成rpc调用
               // 3.1 准备请求参数
               // 填充参数
               HelloProto.HelloRequest.Builder builder = HelloProto.HelloRequest.newBuilder();
               builder.setName("wjfeng");
               HelloProto.HelloRequest helloRequest = builder.build();
               // 3.2 调用rpc服务，获取响应内容
               HelloProto.HelloResponse helloResponse = helloService.hello(helloRequest);
   
               String result = helloResponse.getResult();
               System.out.println("result = " + result);
           } catch (Exception e) {
               throw new RuntimeException(e);
           } finally {
               managedChannel.shutdown();
           }
       }
   }
   ```

### gRPC的主要API

gRPC是一个高性能、跨平台的远程过程调用（RPC）框架，它使用Protocol Buffers作为接口定义语言。gRPC提供了多种API，用于定义和实现RPC服务

以下是gRPC的主要API：

1. Protocol Buffers（Proto）：gRPC使用Protocol Buffers作为接口定义语言（IDL）。Proto API用于定义服务的消息类型和服务接口
2. Server API：gRPC Server API用于创建和管理gRPC服务端。它提供了创建RPC服务、启动和停止服务、配置监听端口等功能
3. Client API：gRPC Client API用于创建和管理gRPC客户端。它提供了与服务端建立连接、发送RPC请求、处理响应等功能
4. Service API：gRPC Service API用于定义服务接口。它基于Proto文件中定义的服务接口生成相应的服务端和客户端代码
5. Interceptors API：gRPC Interceptors API允许开发者拦截和处理gRPC请求和响应。拦截器可用于实现认证、日志记录、性能监控等功能
6. Streaming API：gRPC支持基于流的RPC，其中客户端和服务端可以通过流式传输发送和接收消息。Streaming API包括客户端流、服务端流和双向流
7. Metadata API：gRPC Metadata API用于传递附加的元数据信息。元数据可以在RPC调用中携带额外的信息，如认证凭证、请求标识等
8. Error Handling API：gRPC提供了用于处理错误的API，包括定义错误码和错误信息、处理异常、返回错误状态等

### 生成接口

Protobuf plugin的compile生成message，compile-custom生成service API

message生成的API放在target/generated-sources/protobuf/**java**/java_package/java_outer_classname.java

service生成的API放在target/generated-sources/protobuf/**grpc-java**/java_package/service_nameGrpc.java

<img src="生成的接口.drawio.png">

* `serviceName + Impl + Base` 对应真正的服务接口，开发的时候要继承这个类，并覆盖其中的业务方法
* `Stub` 结尾，用于在客户端与远程 gRPC 服务进行通信，区别在于采用的通信方式不同

### 定义服务 & StreamObserver

虽然service中定义了返回值，但实际实现是返回了void，**通过参数 `StreamObserver` 来返回**，这是一种观察者模式。还有可能通过流等方式来返回，这跟每种stub的底层通信方式有关系

在gRPC中，StreamObserver是一种用于处理流式响应的接口。它在客户端和服务器之间建立了一个双向通信通道，允许客户端和服务器以流式方式交换数据

StreamObserver在gRPC中的主要作用如下

1. **接收服务器流式响应**：当客户端向服务器发起一个请求并期望服务器以流式方式返回多个响应时，客户端可以使用StreamObserver来接收这些响应。StreamObserver提供了一个回调函数，每当服务器发送一个响应时，该函数就会被调用，从而让客户端能够逐个接收和处理这些响应
2. **发送客户端流式请求**：与服务器流式响应相反，客户端可以使用StreamObserver来发送一个流式请求给服务器。客户端可以通过StreamObserver提供的方法来逐个发送请求消息，而不是一次发送所有请求。这种方式使得客户端能够按需发送请求，而不需要等待所有请求准备就绪
3. **实现双向流式通信**：除了单向的服务器流式响应和客户端流式请求之外，gRPC还支持双向流式通信，即客户端和服务器都可以以流式方式发送和接收数据。在这种情况下，StreamObserver既可以接收服务器的响应，也可以发送客户端的请求。这种双向流式通信的模式在需要实时互动和持续通信的场景中非常有用

以下面的unary为例

```java
public class HelloServiceImpl extends HelloServiceGrpc.HelloServiceImplBase {
    // 1. 接受client的参数
    // 2. 业务处理 service+dao 调用对应的业务功能
    // 3. 提供返回值
    @Override
    // service中定义的返回值是作为参数StreamObserver来传递
    public void hello(HelloProto.HelloRequest request, StreamObserver<HelloProto.HelloResponse> responseStreamObserver) {
        // 1. 接受client的参数
        String name = request.getName();
        // 2. 业务处理
        System.out.println("接受到客户端信息:" + name);
        // 3. 封装响应
        // 3.1 构建响应对象
        HelloProto.HelloResponse.Builder builder = HelloProto.HelloResponse.newBuilder();
        // 3.2 填充数据
        builder.setResult("hello method invoke ok");
        // 3.3 封装响应
        HelloProto.HelloResponse helloResponse = builder.build();
        // 3.4
        responseStreamObserver.onNext(helloResponse); // 处理后的响应通过网络回传给client
        responseStreamObserver.onCompleted(); // 通知client 响应已经结束了，会返回一个标志，client接收到这个标志后，会结束这次rpc调用
    }
}
```

## *gRPC的4种通信方式*

### 什么是stub

在 *计算机网络.md* 的RPC协议部分介绍过 RPC 框架的构成，其中 stub 就是一个介于 client/server function 与 kernel 的网络栈之间的位于用户空间的一层软件层。即 stub 存根/代理是client和server之间的接口代理。Stub充当client和server之间的中间人，隐藏了底层的网络通信细节，使得远程过程调用过程对开发者透明化。一言以概之：**stub就是对通信过程（序列化 + 通信方式）的封装**

当client希望调用远程服务器上的方法时，它不会直接与server进行通信，而是通过stub来发送请求。stub负责将请求打包并通过网络发送到server。在服务器端，stub将接收到的请求解包，并将其传递给实际的服务实现，然后将执行结果返回给client

stub的工作原理通常涉及序列化和反序列化过程，它会将方法调用和参数打包成网络传输格式，并在接收方将其解析回原始形式

在许多RPC框架中，stub通常是通过使用接口定义语言（IDL）来生成的。IDL描述了可用的方法和参数，并根据IDL生成相应的客户端和服务器代码。这种自动生成Stub的方式使得客户端和服务器能够在不了解底层网络协议的情况下进行远程方法调用

### 分类

gRPC 支持四种不同的调用方式，满足不同的需求

1. 简单RPC/一元RPC Unary RPC：一个请求对应一个响应
2. 服务端流式RPC Server Streaming RPC：一个请求对应多个响应
3. 客户端流式RPC Client Streaming RPC：多个请求对应一个响应
4. 双向流RPC Bi-directional Stream RPC：多个请求返回多个响应

### gRPC 代理方法 stub

1. BlockingStub 阻塞通信：支持 Unary 和 Server streaming
2. Stub 异步通信，通过监听处理：支持 Unary、Server-streaming、Client-streaming、Bidirectional-streaming
3. FutureStub：FutureStub 只支持 Unary，实际引用比较局限

Stub Asynchronous 和 Future的区别：**Future最适合的场景就是一个大任务需要多个小任务，只有小任务都完成了大任务才能执行**。而Asynchronous 适合的场景就是**多个任务之间并没有顺序关系，都是独立的任务**

### 一元RPC

<img src="一元RPC.drawio.png">

一元 RPC 中，client 向 server 发送单个请求并获得单个响应，就像正常的函数调用一样

client 和 server发送信息后必须要阻塞等待，就是传统Web开发中的请求响应。开发过程中，主要采用就是一元RPC的这种通信方式

```protobuf
service HelloService{
  rpc hello(HelloRequest) returns (HelloResponse)
}
```

## *服务端流式RPC*

<img src="服务端流式RPC.drawio.png">

对于一个请求对象，在不同的时刻Server可以返回多个结果对象。这种服务式基于长连接的

**错误的认知**：认为Server返回的是一组数据就应该封装在一个List中。若把返回的多个数据封装在一个List中，这叫做一个返回结果

应用场景：某一个时段内的股票价格

### protobuf设置

```protobuf
service HelloService{
  rpc hello(HelloRequest) returns (stream HelloResponse) {} // response 加 stream
}
```

### 阻塞demo

一元和服务端流式创建的都是Blocking Stub代理

阻塞：一旦发起服务端流式RPC，Client只有收到所有信息后（读到了 `onCompleted()`）它才会继续执行。实际中这种阻塞通信用处不大

```java
// Server Service
public class HelloServiceImpl extends HelloServiceGrpc.HelloServiceImplBase {
	@Override
    public void c2ss(HelloProto.HelloRequest request,
                     StreamObserver<HelloProto.HelloResponse> responseObserver)  {
        // 1. 接受client的参数
        String name = request.getName();
        // 2. 业务处理
        System.out.println("接受到客户端信息:" + name);
        // 3. 封装响应
        for (int i = 0; i < 9; i++) {
            HelloProto.HelloResponse.Builder builder = HelloProto.HelloResponse.newBuilder();
            builder.setResult("result" + i);
            HelloProto.HelloResponse helloResponse = builder.build();

            responseObserver.onNext(helloResponse);

            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        responseObserver.onCompleted();
    }
}

// Client
public class GrpcClient3 {
    public static void main(String[] args) {
        ManagedChannel managedChannel = ManagedChannelBuilder.forAddress("localhost", 9000).usePlaintext().build();
        try {
            // 用阻塞stub来通信
            HelloServiceGrpc.HelloServiceBlockingStub helloService = HelloServiceGrpc.newBlockingStub(managedChannel);
            HelloProto.HelloRequest helloRequest = HelloProto.HelloRequest.newBuilder().setName("wjfeng").build();
            // 流式的用Iterator作为返回值
            Iterator<HelloProto.HelloResponse> helloResponseIterator = helloService.c2ss(helloRequest);
            while (helloResponseIterator.hasNext()) {
                HelloProto.HelloResponse helloResponse = helloResponseIterator.next();
                System.out.println("result = " + helloResponse.getResult());
            }
        } catch(Exception e) {
            throw new RuntimeException(e);
        } finally {
            managedChannel.shutdown();
        }
    }
}
```

### 异步监听

Api和服务都不变，客户端的调用处理变了

StreamObserver可以监听三种事件 `onNext()` 是下一条信息到来了要干什么，`onError()` 报错了要干什么，`onCompleted()` 事件发完了要干什么

观察者模式编程：api和server不变，改变client的stub，并自定义 `StreamObserver<Response>`的三种方法

```java
//api和server不变，改变client的stub，并自定义StreamObserver<Response>的行为
public class GrpcClient4 {
    public static void main(String[] args) {
        ManagedChannel managedChannel = ManagedChannelBuilder.forAddress("localhost", 9000).usePlaintext().build();
        try {
            // 异步监听
            HelloServiceGrpc.HelloServiceStub helloService = HelloServiceGrpc.newStub(managedChannel);
            HelloProto.HelloRequest.Builder builder=  HelloProto.HelloRequest.newBuilder();
            builder.setName("wjfeng");

            HelloProto.HelloRequest helloRequest = builder.build();
            helloService.c2ss(helloRequest, new StreamObserver<HelloProto.HelloResponse>() {
                @Override
                public void onNext(HelloProto.HelloResponse helloResponse) {
                    // 服务器响应了一个消息后，需要立即处理的话就写这个方法
                    System.out.println("result = " + helloResponse.getResult());
                }

                @Override
                public void onError(Throwable throwable) {
                    System.out.println("error");
                }

                @Override
                public void onCompleted() {
                    // 服务器响应完了所有消息后，需要立即处理的话就写这个方法
                    System.out.println("completed");
                }
            });

            managedChannel.awaitTermination(12, TimeUnit.SECONDS);
        }
        catch(Exception e) {
            throw new RuntimeException(e);
        } finally {
            managedChannel.shutdown();
        }
    }
}
```

存在一个问题：刚开始异步监听时，服务端要首先处理一些逻辑，所以客户端发现没有信息，所以直接关了。等到服务端要发信息过来，发现客户端没开，双方之间也就没有任何通信。所以最好一开始让client等上一段时间

## *客户端流式RPC*

<img src="客户端流式RPC.drawio.png">

应用：IOT传感器

### protobuf设置

```protobuf
service HelloService{
	rpc cs2s(stream HelloRequest) returns (HelloResponse) {}
}
```

### client stream代码组织

<img src="clientStream流程.drawio.png">

server `StreamObserver<Request>` 的 `onCompleted` 中为什么要先发onNext再completed？因为要靠onNext发送信息，completed只是一个结束标志

注意：client stream rpc中server不能每次收到一个request就onNext一个response，**只能是在onCompleted中收到了收到request后再onNext一个response**。否则会报错

```
Cancelling the stream with status Status{code=INTERNAL, description=Too many responses, cause=null}
```


只有在[双向流](#双向流rpc)中可以来一个request就onNext一个response

### server

注意：编译后发现client stream rpc要重写的接口为下面的，之前的unary和server stream的服务接口都是传入request和 `StreamObserver<Response>`，返回void，但现在返回了 `StreamObserver<Request>`

```java
//对比：server stream
public void c2ss(HelloProto.HelloRequest request, StreamObserver<HelloProto.HelloResponse> responseObserver) {}
//client stream
public StreamObserver<HelloProto.HelloRequest> cs2s(StreamObserver<HelloProto.HelloResponse> responseObserver) {}
```

这是因为此时发的是一批request，server不知道它们什么时候才会到，也不知道到没到。所以要用 `StreamObserver<Request>` 监控。因此相应的Server要根据业务需求重写 `StreamObserver<Request>` 里面的方法

```java
public class HelloServiceImpl extends HelloServiceGrpc.HelloServiceImplBase {
	@Override
    //StreamObserver监控的是来的Request
    public StreamObserver<HelloProto.HelloRequest> cs2s(
            StreamObserver<HelloProto.HelloResponse> responseObserver) {
        return new StreamObserver<HelloProto.HelloRequest>() {
            @Override
            public void onNext(HelloProto.HelloRequest value) {
                String name = value.getName();
                System.out.println("Server received one message: name = " + name);
            }

            @Override
            public void onError(Throwable t) {
                System.out.println("onError");
            }

            @Override
            public void onCompleted() {
                System.out.println("Client send completed");
                // 提供响应：接收到了全部的client的请求，提供响应
                HelloProto.HelloResponse.Builder builder = HelloProto.HelloResponse.newBuilder();
                builder.setResult("hello method invoke ok");
                HelloProto.HelloResponse helloResponse = builder.build();
                // 这是处理响应的，注意不要和监控stream的StreamObserver搞混了
                responseObserver.onNext(helloResponse);
                responseObserver.onCompleted();
            }
        };
    }
}
```

### client

```java
public class GrpcClient5 {
    public static void main(String[] args) {
        ManagedChannel managedChannel = ManagedChannelBuilder.forAddress("localhost", 9000).usePlaintext().build();
        try {
            //不能阻塞，要用异步监听
            HelloServiceGrpc.HelloServiceStub helloService = HelloServiceGrpc.
                newStub(managedChannel);
            //监控response
            StreamObserver<HelloProto.HelloRequest> helloRequestStreamObserver = helloService.cs2s(new StreamObserver<HelloProto.HelloResponse>() {
               @Override
               public void onNext(HelloProto.HelloResponse helloResponse) {
                     System.out.println("result = " + helloResponse.getResult());
               }

               @Override
               public void onError(Throwable throwable) {

               }

               @Override
               public void onCompleted() {

               }
            });

            //发送消息
            for (int i= 0; i < 10; i++) {
                HelloProto.HelloRequest helloRequest = HelloProto.HelloRequest.
                    newBuilder().setName("wjfeng" + i).build();
                helloRequestStreamObserver.onNext(helloRequest);

                Thread.sleep((1000));
            }
            helloRequestStreamObserver.onCompleted(); //发完了

            managedChannel.awaitTermination(12, java.util.concurrent.TimeUnit.SECONDS);
            System.out.println("client terminated");
        } catch(Exception e) {
            throw new RuntimeException(e);
        } finally {
            managedChannel.shutdown();
        }
    }
}
```

## *<span id="双向流rpc">双向流RPC</span>*

双向流rpc和client stream 异步stub的代码结构完全一样，区别只是现在server在监控request的时候也可以每次来一个request就回复response了

```java
public class HelloServiceImpl extends HelloServiceGrpc.HelloServiceImplBase {
	@Override
    public StreamObserver<de.tum.HelloProto.HelloRequest> cs2ss(
            StreamObserver<de.tum.HelloProto.HelloResponse> responseObserver) {
        return new StreamObserver<HelloProto.HelloRequest>() {
            @Override
            public void onNext(HelloProto.HelloRequest value) {
                // 现在server可以来一个client的request就回一个response了
                String name = value.getName();
                System.out.println("Server received one message: name = " + name);
                responseObserver.onNext(HelloProto.HelloResponse.newBuilder().
                                        setResult("response " + name + " result ").build());
            }

            @Override
            public void onError(Throwable t) {
                System.out.println("onError");
            }

            @Override
            public void onCompleted() {
                System.out.println("Client send completed");
                responseObserver.onCompleted();
            }
        };
    }
}
```

`addListener` 智能监听，实战中基本没什么用

## *C++ API*

https://grpc.io/docs/languages/cpp/basics/

### 使用流程

1. **定义服务接口和消息类型**：和 Java 一样使用 protobuf 语言定义服务接口和消息类型。比方说下面创建一个 `hello_service.proto` 文件，其中包含一个简单的问候服务的定义

   ```protobuf
   syntax = "proto3";
   
   package com.example;
   
   service HelloService {
       rpc SayHello (HelloRequest) returns (HelloResponse);
   }
   
   message HelloRequest {
       string name = 1;
   }
   
   message HelloResponse {
       string message = 1;
   }
   ```

2. **生成 C++ 代码**：用 protoc 将 `.proto` 文件编译为 C++ 代码。不过生成 service 的 protoc 选项和普通的 message 的不太一样

   ```cmd
   $ protoc -I . --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` ./hello_service.proto
   ```

   生成的文件被命名为 `hello_service.grpc.pb.h` 和  `hello_service.grpc.pb.cc`

   里面包括了一个 HelloService 的 stub 类，以及 `HelloService::Service` 这个需要重写的接口类

3. 创建 server（以最简单的一元服务器为例）

   * **编写 server 的服务接口**：为了实现 `HelloService` 接口，需要创建一个类（一般命名为 `service名 + Impl`，比如说这里叫 `HelloServiceImpl`），并继承自生成的 `grpc::proto::HelloService::Service` 类。在类中，实现服务接口定义的方法

      ```c++
      /*rpc_server.cc*/
      #include <iostream>
      #include <grpcpp/grpcpp.h>
      #include "hello_service.grpc.pb.h"
      
      using grpc::Server;
      using grpc::ServerBuilder;
      using grpc::ServerContext;
      using grpc::Status;
      using hello::HelloRequest;
      using hello::HelloResponse;
      using hello::HelloService;
      
      class HelloServiceImpl final : public grpc::proto::HelloService::Service {
          ::grpc::Status SayHello(ServerContext* context,
                                  const HelloRequest* request,
                                  HelloResponse* response) override {
              std::string name = request->name();
              std::string message = "Hello, " + name + "!";
              response->set_message(message);
              return Status::OK;
          }
      };
      ```

   * **启动 server 的监听和启动服务**

      ```c++
      /*rpc_server.cc*/
      void RunServer(const std::string& db_path) {
        std::string server_address("0.0.0.0:50051");
        RouteGuideImpl service(db_path);
      
        ServerBuilder builder; // 实例化工厂类对象
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
        builder.RegisterService(&service
                                
        std::unique_ptr<Server> server(builder.BuildAndStart());
        std::cout << "Server listening on " << server_address << std::endl;
        server->Wait();
      }
      
      int main() {
          RunServer();
          return 0;
      }
      ```

4. **创建 gRPC 客户端**：首先要创建一个 `Channel` 对象，指定服务器的地址和端口号。然后可以通过使用生成的客户端 stub（例如 `HelloService::NewStub(channel)`）来调用服务方法

   ```cpp
   #include <iostream>
   #include <grpcpp/grpcpp.h>
   #include "hello_service.grpc.pb.h"
   
   using grpc::Channel;
   using grpc::ClientContext;
   using grpc::Status;
   using hello::HelloRequest;
   using hello::HelloResponse;
   using hello::HelloService;
   
   class HelloClient {
   public:
       HelloClient(std::shared_ptr<Channel> channel)
           : stub_(HelloService::NewStub(channel)) {} // 建立 stub
   
       std::string SayHello(const std::string& name) {
           HelloRequest request;
           request.set_name(name);
           HelloResponse response;
   
           ClientContext context;
           Status status = stub_->SayHello(&context, request, &response);
   
           if (status.ok()) {
               return response.message();
           } else {
               return "RPC failed";
           }
       }
   
   private:
       std::unique_ptr<HelloService::Stub> stub_;
   };
   
   int main() {
       std::string server_address("localhost:50051");
       HelloClient client(grpc::CreateChannel(server_address,
                                              grpc::InsecureChannelCredentials()));
       std::string name("Alice");
       std::string response = client.SayHello(name);
       std::cout << "Response: " << response << std::endl;
       return 0;
   }
   ```

### 不同类型的通信方式

* 简单RPC/一元RPC Unary RPC：一个请求对应一个响应
* 服务端流式RPC Server Streaming RPC：一个请求对应多个响应
* 客户端流式RPC Client Streaming RPC：多个请求对应一个响应
* 双向流RPC Bi-directional Stream RPC：多个请求返回多个响应

# HTTP/2 信息处理



## *Metadata*

## *处理 HTTP/2 帧*

## *Compressor*

# 日志

# 管理分布式系统

支持拦截器、服务发现和负载均衡等中间件：gRPC 允许开发者使用拦截器和中间件来添加自定义的逻辑和功能。这包括身份验证、日志记录、错误处理等，提供了更灵活的扩展性和可定制性

## *拦截器*

### 拦截器的功能

当执行远程调用时，无论是 client 还是 server，在远程方法执行之前或之后，都可能需要执行一些通用逻辑。在gRPC 中，最常见的就是可以拦截 RPC 的执行，来满足特定的需求，如日志、身份验证、授权、性能度量指标、跟踪等，这会使用一种名为拦截器 interceptor 的扩展机制

gRPC 提供了简单的 API，用来在 client 和 server 的 gRPC 应用程序中实现并安装拦截器。不过并非 gRPC 所有语言的 API 的实现和功能都是相同的，要具体语言具体分析

gRPC 拦截器可以分为

* 对于一元RPC，可以使用一元拦截器 unary interceptor
* 对于流 RPC，可以使用流拦截器 streaming interceptor

## *服务发现：Resolver*

Resolver 名称解析器负责根据服务名称找到对应的服务器地址，在微服务场景下，对应于同一个服务名称，后端可能部署多个相同的后台服务，Resolver 会找到多服务器地址，通常可以使用etcd、zookeeper等进行服务注册与发现，自实现 Resolver 模块从etcd上读取注册上来的地址列表

### 自定义 Resolver

支持 Java 和 Go，不支持 C++ 和 Python

## *负载均衡：Balance*

Balance 从多个地址中选择一个后端进行访问，gRPC支持 first pick、roundrobin 等负载均衡策略，同时支持自定义Balance

### 自定义 Balance

支持 Java，不支持 Go 和 C++ 

# 中间件





# 安全验证

## *Overview*

### CreateChannel 接口

在创建 client 时的步骤是这样的

```c++
std::string server_address("localhost:50051");
HelloClient client(grpc::CreateChannel(server_address,
                                       grpc::InsecureChannelCredentials()));
```

具体来说 `grpc::CreateChannel()` 接口的定义为 

```c++
std::shared_ptr<Channel> grpc::CreateChannel(const grpc::string & target,
                                            const std::shared_ptr<ChannelCredentials>& creds)
    
std::shared_ptr<ServerCredentials> grpc::InsecureServerCredentials();
```

其中第二个参数是一个智能指针，它管理一个 ChannelCredentials 的类实例，这个类表示的就是不同的安全验证。默认调用的就是 `InsecureServerCredentials()`，即不安全的、未加密的通道

### gRPC内置的安全验证方式

gRPC 设计用于与各种身份验证机制配合使用，使得通过 gRPC 安全地与其他系统通信变得容易

* SSL/TLS：gRPC 具有 SSL/TLS 集成，并推广使用 SSL/TLS 来验证服务器，并用来加 client 和 server 之间交换的所有数据。可选机制可供客户端提供证书进行相互身份验证

* ALTS：如果应用程序在计算引擎或 Google Kubernetes Engine（GKE）上运行，则 gRPC 支持 ALTS 作为传输安全机制

  Application Layer Transport Security, ALTS 是由 Google 所开发的双向身份验证和传输加密系统，通常用于在*Google* 基础架构内部保护 RPC 通信的安全

  支持 C++、GO、Java、Python

* 基于 Google 的令牌身份验证 / JWT：gRPC 提供了一种通用机制（如下所述），用于将基于元数据的凭据附加到请求和响应上。在通过 gRPC 访问 Google API 时还提供了获取访问令牌（通常是 OAuth2 令牌）的额外支持

gRPC也支持来用户使用 API 来接入用户自定义的身份验证系统

## *Basic 认证*

### JWT 认证

JSON Web Token, JWT

### TLS 认证

https://blog.csdn.net/chenwr2018/article/details/105708168

```c++
// Create a default SSL ChannelCredentials object.
auto channel_creds = grpc::SslCredentials(grpc::SslCredentialsOptions());
// Create a channel using the credentials created in the previous step.
auto channel = grpc::CreateChannel(server_name, channel_creds);
// Create a stub on the channel.
std::unique_ptr<Greeter::Stub> stub(Greeter::NewStub(channel));
// Make actual RPC calls on the stub.
grpc::Status s = stub->sayHello(&context, *request, response);
```



